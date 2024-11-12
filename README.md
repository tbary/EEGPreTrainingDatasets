# Generation of Pre-training Datasets for EEG Classification

This repository contains the code used for my master’s thesis at UCLouvain. The objective is to generate several pre-training datasets from EEG data and evaluate their effectiveness on the MViT transformer architecture. For further details on the methodology, refer to Annex A of my master thesis, which is included in this repository.

## Requirements

To run the code in this repository, please install the following packages:

- `pandas` 1.4.2
- `numpy` 1.24.3
- `pywt` 1.5.0 ([PyWavelets GitHub](https://github.com/PyWavelets/pywt))
- `matplotlib` 3.7.1
- `tensorflow` 2.10.0
- `tensorflow_addons` 0.18.0
- `scipy` 1.8.0
- `pyedflib` 0.1.30
- `tqdm` 4.64.1
- `scikit-learn` 1.1.3

*Note:* These package versions were used in development. Other versions may also work but are untested.

## Datasets

The datasets used in this project cannot be publicly shared due to licensing restrictions. One dataset is proprietary to UCLouvain, while access to the other dataset requires an online form submission.

The scripts `4_pre_processing_TUSZ.py`, `5_train_model_TUSZ.py`, and `6_TUSZ_model_performances.py` are compatible with `.edf` files. If starting the pipeline from EEG data, begin with `4_pre_processing_TUSZ.py`.

For access to the Temple University Seizure Detection Corpus (TUSZ) used in this project, please fill out the form [here](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).

## Citation

This work was presented at the 22nd IEEE Mediterranean Electrotechnical Conference (MELECON 2024) ([link to paper](https://arxiv.org/abs/2410.07190)).

If you use this code in your own research, please cite as follows:

```tex
@inproceedings{bary2024designing,
  title={Designing Pre-training Datasets from Unlabeled Data for EEG Classification with Transformers},
  author={Bary, Tim and Macq, Beno{\^\i}t},
  booktitle={22nd IEEE Mediterranean Electrotechnical Conference (MELECON 2024)},
  year={2024}
}

