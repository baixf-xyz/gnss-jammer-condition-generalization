#!/usr/bin/env python3

import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib.pyplot as plt

from wespeaker.frontend.rf_stft import RFSTFTFrontend
from wespeaker.frontend.rf_complex_stft import RFComplexSTFTFrontend
from wespeaker.frontend.rf_mag_phasediff import RFMagPhaseDiffFrontend
from wespeaker.models.resnet import ResNet34


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--rf_num_samples", type=int, default=16384)
    parser.add_argument(
        "--num_crops",
        type=int,
        default=1,
        help="Number of deterministic temporal crops per RF record."
    )

    parser.add_argument(
        "--frontend",
        choices=["rf_stft", "rf_complex_stft", "rf_mag_phasediff"],
        default="rf_stft",
    )

    parser.add_argument(
        "--phase_deg",
        type=float,
        default=0.0,
        help="Apply global IQ phase rotation in degrees before frontend."
    )

    return parser.parse_args()


def load_label_map(path):
    """
    train_label format:
        sample_key class_name
    """

    classes = set()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            label = parts[1]
            classes.add(label)

    classes = sorted(classes)

    label2id = {
        label: idx
        for idx, label in enumerate(classes)
    }

    return label2id

class RFH5Dataset(Dataset):

    def __init__(
        self,
        manifest,
        label2id,
        rf_num_samples=16384,
        num_crops=1,
    ):
        self.items = []

        with open(manifest, "r") as f:
            for line in f:
                line = line.strip()

                if line:
                    self.items.append(json.loads(line))

        self.label2id = label2id
        self.rf_num_samples = rf_num_samples
        self.num_crops = num_crops

        # each dataloader worker gets its own cache
        self._h5_cache = {}

    def __len__(self):
        return len(self.items)

    def _get_h5(self, path):
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(
                path,
                "r",
            )

        return self._h5_cache[path]

    def __getitem__(self, idx):

        obj = self.items[idx]

        h5_path = obj["h5"]
        row = int(obj["row"])

        f = self._get_h5(h5_path)

        raw = np.asarray(
            f["data"][row],
            dtype=np.float32,
        )

        # interleaved IQ:
        # I0 Q0 I1 Q1 ...
        i = raw[0::2]
        q = raw[1::2]

        iq = np.stack(
            [i, q],
            axis=0,
        )

        n = iq.shape[-1]
        target_n = self.rf_num_samples

        # -------------------------------------------------
        # Deterministic multi-crop evaluation
        #
        # num_crops = 1:
        #   center crop
        #
        # num_crops = 5:
        #   evenly spaced crops from start to end
        #
        # output:
        #   [K, 2, target_n]
        # -------------------------------------------------
        if n >= target_n:

            max_start = n - target_n

            if self.num_crops <= 1:
                starts = [max_start // 2]
            else:
                starts = np.linspace(
                    0,
                    max_start,
                    self.num_crops,
                    dtype=int,
                ).tolist()

            crops = []

            for start in starts:

                crop = iq[
                    :,
                    start:start + target_n
                ].copy()

                # RMS normalization per crop
                rms = np.sqrt(
                    np.mean(crop ** 2) + 1e-8
                )

                crop = crop / rms

                crops.append(crop)

            iq = np.stack(
                crops,
                axis=0,
            )

        else:

            pad = target_n - n

            crop = np.pad(
                iq,
                ((0, 0), (0, pad)),
                mode="constant",
            )

            rms = np.sqrt(
                np.mean(crop ** 2) + 1e-8
            )

            crop = crop / rms

            iq = crop[None, ...]

        x = torch.from_numpy(
            iq.astype(np.float32)
        )

        label_name = obj["spk"]
        y = self.label2id[label_name]

        return x, y, obj["key"]


def get_state_dict(ckpt):

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            "Unexpected checkpoint format."
        )

    # Common WeSpeaker / PyTorch formats
    for candidate in [
        "model",
        "state_dict",
        "model_state_dict",
    ]:
        if (
            candidate in ckpt
            and isinstance(ckpt[candidate], dict)
        ):
            state = ckpt[candidate]
            break
    else:
        # Direct state_dict
        if all(
            torch.is_tensor(v)
            for v in ckpt.values()
        ):
            state = ckpt
        else:
            # Search nested dict automatically
            state = None

            for k, v in ckpt.items():
                if (
                    isinstance(v, dict)
                    and any(
                        torch.is_tensor(x)
                        for x in v.values()
                    )
                ):
                    print(
                        f"Using nested checkpoint key: {k}"
                    )
                    state = v
                    break

            if state is None:
                raise RuntimeError(
                    "Cannot locate model state_dict."
                )

    cleaned = {}

    for k, v in state.items():

        # strip DDP prefix
        if k.startswith("module."):
            k = k[len("module."):]

        cleaned[k] = v

    return cleaned


def find_projection_weight(state):

    candidates = []

    for k, v in state.items():
        if (
            "projection" in k
            and k.endswith("weight")
            and torch.is_tensor(v)
            and v.ndim == 2
        ):
            candidates.append((k, v))

    if not candidates:
        print("\nAvailable projection-related keys:")

        for k in state:
            if "projection" in k:
                print(" ", k)

        raise RuntimeError(
            "Cannot find projection weight."
        )

    print("\nProjection candidates:")

    for k, v in candidates:
        print(
            " ",
            k,
            tuple(v.shape),
        )

    # Normally only one
    return candidates[0]


def main():

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # -------------------------------------
    # Labels
    # -------------------------------------

    label2id = load_label_map(
        args.label_file
    )

    id2label = {
        v: k
        for k, v in label2id.items()
    }

    print("\nLabel map:")

    for k, v in sorted(
        label2id.items(),
        key=lambda x: x[1]
    ):
        print(v, k)

    num_classes = len(label2id)

    # -------------------------------------
    # Data
    # -------------------------------------

    dataset = RFH5Dataset(
        args.manifest,
        label2id,
        rf_num_samples=args.rf_num_samples,
        num_crops=args.num_crops,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        "\nEvaluation samples:",
        len(dataset),
    )

    print(
        "Crops per sample:",
        args.num_crops,
    )

    # -------------------------------------
    # Model
    # -------------------------------------

    if args.frontend == "rf_stft":

        frontend = RFSTFTFrontend(
            n_fft=512,
            hop_length=128,
            win_length=512,
            log_scale=True,
            normalize=True,
            center=False,
        ).to(device)

        feat_dim = 512

    elif args.frontend == "rf_complex_stft":

        frontend = RFComplexSTFTFrontend(
            n_fft=512,
            hop_length=128,
            win_length=512,
            normalize=True,
            center=False,
        ).to(device)

        feat_dim = 1024

    elif args.frontend == "rf_mag_phasediff":

        frontend = RFMagPhaseDiffFrontend(
            n_fft=512,
            hop_length=128,
            win_length=512,
            normalize=True,
            center=False,
        ).to(device)

        feat_dim = 1024

    else:
        raise ValueError(args.frontend)

    print("Frontend:", args.frontend)
    print("Feature dim:", feat_dim)
    print("Global phase rotation:", args.phase_deg, "deg")

    model = ResNet34(
        feat_dim=feat_dim,
        embed_dim=192,
        pooling_func="TSTP",
        two_emb_layer=False,
    )

    model = model.to(device)

    # -------------------------------------
    # Checkpoint
    # -------------------------------------

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    state = get_state_dict(ckpt)

    proj_key, proj_weight = (
        find_projection_weight(state)
    )

    print(
        "\nUsing projection:",
        proj_key,
        tuple(proj_weight.shape),
    )

    # Remove components not required by
    # backbone model loading.
    backbone_state = {}

    for k, v in state.items():

        if k.startswith("projection."):
            continue

        if k.startswith("frontend."):
            continue

        backbone_state[k] = v

    result = model.load_state_dict(
        backbone_state,
        strict=False,
    )

    print(
        "\nMissing keys:",
        result.missing_keys,
    )

    print(
        "Unexpected keys:",
        result.unexpected_keys,
    )

    proj_weight = proj_weight.to(device)

    model.eval()
    frontend.eval()

    all_true = []
    all_pred = []
    all_keys = []

    # -------------------------------------
    # Evaluation
    # -------------------------------------

    with torch.no_grad():

        for batch_idx, (iq, y, keys) in enumerate(loader):

            iq = iq.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            # DataLoader output:
            #   [B, K, 2, N]
            B, K, C, N = iq.shape

            iq = iq.reshape(
                B * K,
                C,
                N,
            )

            # -------------------------------------------------
            # Global IQ phase rotation:
            #
            # I' = I cos(phi) - Q sin(phi)
            # Q' = I sin(phi) + Q cos(phi)
            #
            # Class identity should ideally be invariant to this.
            # -------------------------------------------------
            if args.phase_deg != 0.0:

                import math

                phi = math.radians(args.phase_deg)

                c = math.cos(phi)
                s_phi = math.sin(phi)

                i0 = iq[:, 0, :].clone()
                q0 = iq[:, 1, :].clone()

                iq[:, 0, :] = c * i0 - s_phi * q0
                iq[:, 1, :] = s_phi * i0 + c * q0

            lens = torch.full(
                (iq.size(0),),
                iq.size(-1),
                dtype=torch.long,
                device=device,
            )

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):

                features, _ = frontend(
                    iq,
                    lens,
                )

                outputs = model(features)

                if isinstance(outputs, tuple):
                    embeddings = outputs[-1]
                else:
                    embeddings = outputs

                embeddings = F.normalize(
                    embeddings.float(),
                    dim=1,
                )

                w = proj_weight.float()

                # ArcMargin projection weight
                # can be stored either
                # [C,D] or [D,C].
                if w.shape[1] == embeddings.shape[1]:

                    w = F.normalize(
                        w,
                        dim=1,
                    )

                    logits = F.linear(
                        embeddings,
                        w,
                    )

                elif w.shape[0] == embeddings.shape[1]:

                    w = F.normalize(
                        w,
                        dim=0,
                    )

                    logits = embeddings @ w

                else:
                    raise RuntimeError(
                        "Projection shape does not "
                        "match embedding dimension: "
                        f"{tuple(w.shape)} vs "
                        f"{tuple(embeddings.shape)}"
                    )

            # -------------------------------------------------
            # Aggregate crop predictions
            #
            # logits:
            #   [B*K, num_classes]
            #
            # -> [B, K, num_classes]
            # -> mean over K
            # -------------------------------------------------
            logits = logits.reshape(
                B,
                K,
                -1,
            )

            logits = logits.mean(
                dim=1
            )

            pred = logits.argmax(
                dim=1
            )

            all_true.extend(
                y.cpu().tolist()
            )

            all_pred.extend(
                pred.cpu().tolist()
            )

            all_keys.extend(keys)

            if (
                batch_idx == 0
                or (batch_idx + 1) % 10 == 0
            ):
                print(
                    f"batch "
                    f"{batch_idx + 1}/"
                    f"{len(loader)}"
                )

    # -------------------------------------
    # Metrics
    # -------------------------------------

    y_true = np.asarray(
        all_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        all_pred,
        dtype=np.int64,
    )

    acc = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    labels = list(range(num_classes))

    target_names = [
        id2label[i]
        for i in labels
    ]

    print("\n" + "=" * 60)

    print(
        f"Accuracy  : {acc * 100:.3f}%"
    )

    print(
        f"Macro F1  : {macro_f1 * 100:.3f}%"
    )

    print("=" * 60)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification report:\n")

    print(report)

    # Save report
    with open(
        os.path.join(
            args.output_dir,
            "metrics.txt",
        ),
        "w",
    ) as f:

        f.write(
            f"checkpoint: "
            f"{args.checkpoint}\n"
        )

        f.write(
            f"accuracy: {acc:.8f}\n"
        )

        f.write(
            f"macro_f1: {macro_f1:.8f}\n\n"
        )

        f.write(report)

    # -------------------------------------
    # Prediction CSV
    # -------------------------------------

    pred_path = os.path.join(
        args.output_dir,
        "predictions.csv",
    )

    with open(
        pred_path,
        "w",
    ) as f:

        f.write(
            "key,true_id,true_label,"
            "pred_id,pred_label\n"
        )

        for key, yt, yp in zip(
            all_keys,
            y_true,
            y_pred,
        ):

            f.write(
                f"{key},"
                f"{yt},"
                f"{id2label[int(yt)]},"
                f"{yp},"
                f"{id2label[int(yp)]}\n"
            )

    # -------------------------------------
    # Confusion matrix
    # -------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    np.savetxt(
        os.path.join(
            args.output_dir,
            "confusion_matrix.csv",
        ),
        cm,
        delimiter=",",
        fmt="%d",
    )

    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(111)

    im = ax.imshow(cm)

    ax.set_title(
        "RF Jammer Classification "
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_xticks(labels)
    ax.set_yticks(labels)

    ax.set_xticklabels(
        target_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        target_names
    )

    for i in range(
        cm.shape[0]
    ):
        for j in range(
            cm.shape[1]
        ):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        im,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            args.output_dir,
            "confusion_matrix.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    print(
        "\nSaved to:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
