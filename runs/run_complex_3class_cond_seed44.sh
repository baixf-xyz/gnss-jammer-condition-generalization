#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1

stage=1
stop_stage=2

data=data/gnss_jammer_3class_cond

config=conf/rf_resnet_complex_3class_cond_seed44.yaml

exp_dir=exp/rf_resnet34_complex_3class_cond_seed44

gpus="[0]"

num_avg=5


# -------------------------------------------------
# Stage 1: Check data
# -------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    echo "========================================"
    echo "Check RF manifests"
    echo "========================================"

    wc -l \
        "${data}/train.list" \
        "${data}/val.list" \
        "${data}/test.list"

    echo

    echo "Train class distribution:"

    awk '{print $2}' \
        "${data}/train_label" \
        | sort \
        | uniq -c

    echo

    echo "First train item:"

    head -1 "${data}/train.list"
fi


# -------------------------------------------------
# Stage 2: Train ResNet34
# -------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo
    echo "========================================"
    echo "Start RF ResNet34 training"
    echo "========================================"

    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=1 \
        ../../wespeaker/bin/train.py \
        --config "${config}" \
        --exp_dir "${exp_dir}" \
        --gpus "${gpus}" \
        --num_avg "${num_avg}" \
        --data_type rf_h5 \
        --train_data "${data}/train.list" \
        --train_label "${data}/train_label"
fi