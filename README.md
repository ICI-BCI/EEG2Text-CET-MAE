# Contrastive Pretraining for EEG-to-Text Generation

This is the official implementation of our IEEE TNSRE paper [Aligning Semantic in Brain and Language: A Curriculum Contrastive Method for Electroencephalography-to-Text Generation](https://ieeexplore.ieee.org/abstract/document/10248031).
Major codes are borrowed from [AAAI 2022 EEG-To-Text](https://github.com/MikeWangWZHL/EEG-To-Text).

## Setup the Environment
You can first setup a conda python environment to make everything works well and then setup the working environment.
```bash
conda create -n eeg2text python=3.9
conda activate eeg2text
cd PATH_TO_PROJECT/contrastive_eeg2text
pip install -r requirements.txt
```

## Data Preparation

### Download Datasets
Following [AAAI 2022 EEG-To-Text](https://github.com/MikeWangWZHL/EEG-To-Text), we employ ZuCo benchmark as our test bench.

1. Download ZuCo v1.0 'Matlab files' for 'task1-SR','task2-NR','task3-TSR' from [Zurich Cognitive Language Processing Corpus: A simultaneous EEG and eye-tracking resource for analyzing the human reading process](https://osf.io/q3zws/files/) under 'OSF Storage' root, unzip and move all .mat files to `./zuco_dataset/task1-SR/Matlab_files`, `./zuco_dataset/task2-NR/Matlab_files`, `./zuco_dataset/task3-TSR/Matlab_files` respectively.
2. Download ZuCo v2.0 'Matlab files' for 'task1-NR' from [ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation](https://osf.io/2urht/files/) under 'OSF Storage' root, unzip and move all .mat files to `./zuco_dataset/task2-NR-2.0/Matlab_files`.

### Preprocess Datasets
```bash
python data_factory/data2pickle_v1.py
python data_factory/data2pickle_v2.py
```


## Contrastive Pretraining

```bash
cd contrastive_eeg_pretraining
python contrastive_train_curriculum.py -c contrastive_config/contrastive_train.yaml
```

## Overall Training
```bash
cd contrastive_eeg2text
python train.py -c config/train.yaml
```


## Testing
```bash
cd contrastive_eeg2text
python eval.py -c config/train.yaml
```



   