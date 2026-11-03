# GNSS Jammer Condition-Disjoint Generalization

Code and results for GNSS interference classification under unseen physical conditions.

## Representations

- Magnitude STFT
- Complex STFT
- Magnitude + temporal phase difference

## Model

- ResNet34
- Embedding dimension: 192
- TSTP pooling
- 3 training epochs
- Seeds: 42, 43, 44

## Condition-disjoint benchmark

Classes:

- Prn
- Meac
- Spoof

Train, validation, and test sets contain no identical physical condition tuples.

Condition tuple:

(class, power, bandwidth, VGA, band)

## Main results

| Representation | Accuracy | Macro-F1 |
|---|---:|---:|
| Magnitude | 60.574 +/- 11.635% | 61.356 +/- 11.236% |
| Complex | 53.093 +/- 3.436% | 51.500 +/- 9.294% |
| Mag + DeltaPhase | 51.333 +/- 3.266% | 52.558 +/- 3.278% |

## Repository structure

- `configs/` experiment configurations
- `runs/` training scripts
- `scripts/` data split, evaluation, and analysis
- `results/` aggregated CSV results
- `figures/` paper figures
- `patches/` WeSpeaker modifications

## Notes

Raw H5 datasets and model checkpoints are not included.

Validation is used for checkpoint selection only. Test results are not used for model selection or benchmark redesign.
