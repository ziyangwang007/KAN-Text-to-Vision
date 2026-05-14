# KANMultiSign

Official implementation of **KANMultiSign: Multi-Scale Sequence-Based Pose Animation from Sign Language Notation with Kolmogorov-Arnold Networks**, accepted by *Neurocomputing*.

KANMultiSign generates 2D sign-language pose sequences from HamNoSys notation. This repository builds on the original [Ham2Pose](https://github.com/rotem-shalev/Ham2Pose) codebase and adds the main components introduced in our Neurocomputing paper, including multi-scale coarse-to-fine supervision and KAN-based Transformer feed-forward layers.

## Overview

Given a HamNoSys sequence, KANMultiSign predicts a corresponding sequence of human poses. The model follows a notation-to-pose generation pipeline:

1. tokenize HamNoSys symbols;
2. encode the symbolic sequence with a text encoder;
3. predict sequence length;
4. iteratively refine the generated pose sequence;
5. use a multi-scale pathway to first learn a coarse 25-part skeleton and then generate the final 137-keypoint pose sequence.

Main features:

- HamNoSys-conditioned sign language pose generation;
- multi-scale coarse-to-fine supervision;
- KAN-based Transformer feed-forward modules;
- support for OpenPose-style 137-keypoint pose sequences;
- DTW-MJE and nDTW-MJE evaluation utilities;
- qualitative pose animation generation.

## Repository Structure

```text
.
├── args.py                         # Command-line arguments and default model settings
├── train.py                        # Training script
├── test.py                         # Evaluation script
├── predict.py                      # Pose prediction and visualization utilities
├── model.py                        # KANMultiSign / Ham2Pose-style model definition
├── KanTransformerEncorder.py       # KAN-based Transformer encoder components
├── kan.py                          # KAN module implementation
├── fasterkan.py                    # FasterKAN implementation
├── metrics.py                      # Evaluation metrics
├── pose_utils.py                   # Pose normalization and visualization helpers
├── calculate_DTW_by_dataset.py     # DTW-MJE calculation utilities
├── interpret_kan.py                # KAN interpretability/visualization script
├── configs/                        # YAML configuration files for ablations and variants
├── data/                           # Dataset loading and preprocessing utilities
├── data_preprocess/                # Data preprocessing and visualization scripts
└── gen_example/                    # Example generation scripts
```

## Installation

We recommend using a clean conda environment.

```bash
conda create -n kanmultisign python=3.7 -y
conda activate kanmultisign
pip install -r requirements.txt
```

If you encounter protobuf-related issues with TensorFlow or `sign_language_datasets`, try:

```bash
pip install --force-reinstall protobuf==3.20.3
```

## Data Preparation

KANMultiSign expects HamNoSys notation paired with 2D pose sequences. In our experiments, each frame contains 137 OpenPose keypoints:

- 25 body keypoints;
- 70 face keypoints;
- 42 hand keypoints, with 21 keypoints per hand.

A typical data preparation workflow is:

1. Download or prepare sign-language videos with HamNoSys annotations.
2. Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract body, face, and hand keypoints.
3. Convert the extracted keypoints into the expected pose format.
4. Place processed files under the expected data directory, or load them through the dataset utilities in `data/`.

The original paper evaluates on sign-language resources covering PJM, DGS, GSL, and LSF. Please check the licence and redistribution terms of each dataset before releasing processed data.

### Important note on preprocessed pickles

The current `train.py`/`test.py` version may expect preprocessed pickle files such as:

```text
temp/train/test_dataset_old.pkl
```

If you do not have these files, you need to generate them from the dataset pipeline first. The commented sections in `train.py` show how the dataset can be created through `get_dataset(...)` and saved as pickle files. Adjust these paths according to your local setup.

## Training

Train the default model:

```bash
python train.py
```

Train with a specific configuration file:

```bash
python train.py --config_file configs/multi-scale-2text4kanpose-64-kan-layer-linearprojection.yaml
```

Useful arguments include:

```bash
python train.py \
  --model_name kanmultisign \
  --num_gpus 1 \
  --max_epochs 2000 \
  --hidden_dim 128 \
  --text_encoder_depth 2 \
  --pose_encoder_depth 4 \
  --text_encoder_type transformer \
  --text_pose_encoder_type kan \
  --pose_projection_type linear \
  --model_variant multiscale
```

The default configuration uses:

- hidden dimension: `128`;
- text encoder depth: `2`;
- text-pose encoder depth: `4`;
- refinement steps: `10`;
- teacher forcing probability: `0.5`;
- sequence length loss weight: `2e-5`;
- optimizer: Adam;
- random seed: `42`.

## Evaluation

Evaluate a trained checkpoint:

```bash
python test.py --model_name kanmultisign --ckpt checkpoints
```

By default, checkpoints are expected under:

```text
models/<model_name>/<ckpt>/model.ckpt
```

The evaluation scripts include support for:

- sequence length prediction error;
- DTW-MJE and nDTW-MJE;
- distance-rank metrics;
- generated pose visualization.

## Prediction and Visualization

Prediction and visualization utilities are implemented in `predict.py`. Generated videos are saved under the model output directory, for example:

```text
models/<model_name>/videos/
```

To visualize intermediate refinement steps, use the `vis_process=True` option in the `pred(...)` function.

## Configuration Files

Several YAML files are provided under `configs/`. Common variants include:

```text
configs/baseline.yaml
configs/multiscale_kan.yaml
configs/multi-scale-supervison.yaml
configs/multi-scale-2text4kanpose-64-kan-layer-linearprojection.yaml
configs/multi-scale-2text6kanpose-64-kan-layer-linearprojection.yaml
configs/multi-scale-2text8kanpose-64-kan-layer-linearprojection.yaml
```

The naming convention is roughly:

- `multi-scale`: uses the coarse-to-fine multi-scale architecture;
- `2text4kanpose`: 2 text-encoder layers and 4 KAN-based text-pose encoder layers;
- `linearprojection`: uses a linear final pose projection;
- `kanprojection`: uses a KAN-based final pose projection;
- `64-kan-layer`: uses a compact KAN hidden dimension.

## Citation

If you use this repository, please cite our paper:

```bibtex
@article{du2026kanmultisign,
  title   = {KANMultiSign: Multi-scale sequence-based pose animation from sign language notation with Kolmogorov-Arnold networks},
  author  = {Du, Guanyi and Wang, Lintao and Hu, Kun and Wang, Ziyang},
  journal = {Neurocomputing},
  year    = {2026},
  doi     = {10.1016/j.neucom.2026.133930}
}
```

This work builds on Ham2Pose. Please also cite the original Ham2Pose paper when using this codebase:

```bibtex
@inproceedings{shalevarkushin2023ham2pose,
  title     = {Ham2Pose: Animating Sign Language Notation into Pose Sequences},
  author    = {Shalev-Arkushin, Rotem and Moryossef, Amit and Fried, Ohad},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {21046--21056},
  year      = {2023}
}
```

## Acknowledgements

This repository is based on the original Ham2Pose implementation by Rotem Shalev-Arkushin, Amit Moryossef, and Ohad Fried. We sincerely thank the authors for releasing their code and establishing a strong baseline for HamNoSys-to-pose generation.

## Licence

Please follow the licence terms of the original Ham2Pose repository and the licences of all third-party dependencies and datasets. If you release this repository publicly, we recommend adding a clear `LICENSE` file before publication.
