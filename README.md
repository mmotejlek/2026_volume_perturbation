# Acoustic Volume Mismatch in Distant Streaming ASR: Case Study and Mitigation via Wide-Range Volume Perturbation

Supplementary content for a TSD 2026 article.

This consists of two repositories:
- [**2026_volume_perturbation**](https://github.com/2026-volume-perturbation/2026_volume_perturbation) – The main repository.
- [**2026_volume_perturbation_wenet**](https://github.com/2026-volume-perturbation/2026_volume_perturbation_wenet) – The modified WeNet code. The repository contains separate `train` and `test` branches, since we patched the code in different ways for each phase. Both branches are included as submodules of the main repository under `wenet_train` and `wenet_test`. **You can view our changes by looking at the latest commits authored by *2026 Volume Perturbation*.**

The following text provides supplementary information and guides for reproduction with the help of the supplied scripts.
Reading the article first is recommended.

**Contents:**

- [Download \& Dependencies](#download--dependencies)
  - [Dependencies](#dependencies)
- [Training](#training)
  - [Training Data](#training-data)
  - [Hyperparameters](#hyperparameters)
  - [Hardware Configuration](#hardware-configuration)
  - [Running the Training](#running-the-training)
- [Testing](#testing)
  - [Test Data](#test-data)
  - [Running the Tests](#running-the-tests)

## Download & Dependencies

To clone the main repository including the submodules with the `train` and `test` WeNet branches, run:

```sh
git clone --recurse-submodules https://github.com/2026-vol-perturb/2026_volume_perturbation
```

Extract the directory from `train_data_lists.zip`.

### Dependencies

**For WeNet (both the `train` and `test` branch of WeNet have the same requirements):**

Python 3.11

```sh
conda install conda-forge::sox
pip install torch==2.6 torchaudio==2.6 --index-url https://download.pytorch.org/whl/cu126
pip install -r wenet_train/requirements.txt
```

**For the scripts in this repository:**

Run this inside the repository.
The latter command installs the `2026_volume_perturbation` package

```sh
pip install -r requirements.txt
pip install -e .
```

## Training

### Training Data

For training, we used the **GigaSpeech** and **English Common Voice** datasets.

With these, we formed the following sets:

- 11800 h (full)
  - train
  - cv
- 500 h
  - train
  - cv

The folder `train_data_lists` contains a **list of files** from each of the source datasets **included in each set**.

**GigaSpeech**:
- The files come from the XL training set (i.e., the entire training set) and the dev set.
- We did not honor the original train/dev split of GigaSpeech and used some of the data from the dev set for training (i.e., in the *train* subset).

**Common Voice**:
- We used a custom selection of data.
- It is possible that the currently available version of Common Voice does not contain all of the files we specify.

We converted the recordings from both datasets to the `opus` format.

The **transcripts** can be generated using the scripts in `transcription_scripts/train` (see the main block at the bottom of each script and the doc comments of the `generate_transcripts` functions).

*Note regarding sample selection:*
We omitted samples that contained abbreviations in their transcripts.
Regarding Common Voice, we chose files based on their rating.
We filtered out sentences where punctuation could no be safely stripped (e.g., due to apostrophe / quotation mark ambiguity) and sentences automatically detected as foreign language.

### Hyperparameters

The folder `training/configs` contains WeNet configuration files for each experiment, which describe most training hyperparameters.
Some settings can be also found in the `.sh` scripts in the `training` directory.

The `Kaldi-like VP`, `proposed VP` and `BVP` models were trained by finetuning from the 45th epoch of baseline for 15 epochs. See the instructions in [Running the Training](#running-the-training) for how this was performed.

### Hardware Configuration

We used the following hardware configurations for training.

**500 h:**

- CPU: 8 cores of AMD EPYC 9454
- GPU: 1x NVIDIA L40S 48GB
- RAM: 100 GB

Time:
- From scratch (60 epochs): about 16 h
- Fine-tuning (15 epochs): about 4 h

**11800 h:**

- CPU: 14 cores of AMD EPYC 9454
- GPU: 2x NVIDIA L40S 48GB
- RAM: 250 GB

Time:
- From scratch (60 epochs): about 216 h
- Fine-tuning (15 epochs): about 54 h

Look for the `NCPUS` and `NGPUS` variables at the top of the provided `.sh` scripts.

### Running the Training

The following text are instructions for training the models with the help of the supplied scripts.

The training takes place in the following directory structure, whose preparation will be described step-by-step:

- `<current working directory>`
    - `wenet_train` – symlink to `wenet_train`
    - `train`
        - `text`
        - `wav.scp`
    - `cv`
        - `text`
        - `wav.scp`
    - `prepare.sh` – copy from `training`
    - `train.sh` – copy from `training`
    - `config.yaml` – copy from `training/configs`; depends on the trained variant
    - `alter_epoch.py` – copy from `training/configs`; depends on the trained variant; only for fine-tuning
    - `lang` – generated folder
    - `global_cmvn` – generated file
    - `train_outs` – the training outputs
      - `epoch_XX.pt` – checkpoint
      - `epoch_XX.yaml` – checkpoint configuration
      - `avg15.pt` – the final model

#### Step 1

1. Create an empty directory that will serve as the working directory.
2. Add a symlink to `wenet_train` into it.

#### Step 2

Prepare the training data as described in [Training Data](#training-data).

Choose either a 11800 h or a 500 h variant of the training set.

Add the following files describing the corresponding `train` and `cv` sets to the directory:

- `train`
    - `text`
    - `wav.scp`
- `cv`
    - `text`
    - `wav.scp`

*Reminder:* The `train` and `cv` sets include files as specified in `train_data_lists`.

The `text` files contain transcripts and have the format:

```
id1 THIS IS A TRANSCRIPT OF THE FIRST FILE
id2 THIS IS A TRANSCRIPT OF THE SECOND FILE
```

The scripts in `transcription_scripts/train` output files with the same format.
Therefore, the `text` files can be obtained by running these scripts and concatenating the outputs for GigaSpeech and Common Voice.

Example: `train/text` for the *500 h* dataset can be obtained by running

- `transcription_scripts/train/gigaspeech.py` with `list_file` pointing to `train_data_lists/500h/train/gigaspeech.txt`,
- `transcription_scripts/train/common_voice.py` with `list_file` pointing to `train_data_lists/500h/train/common_voice.txt`,

and concatenating the outputs.

The `wav.scp` files contain file paths and have the format:

```
id1 path/to/the/first/recording
id2 path/to/the/second/recording
```

The ids match the ids in the `text` files.
We do not provide scripts to generate the `wav.scp` files.

#### Step 3

##### Variant A: `lang` and `global_cmvn` for the 11800 h / 500 h variant have not yet been generated

1. Copy a `config.yaml` from `training/configs/*/baseline` corresponding to the 11800 h or 500 h baseline and rename it to `baseline_config.yaml`.
2. Copy `training/prepare.sh` into the directory.
3. Optionally raise the `NCPUS` specified in the `prepare.sh` script.
4. Run the `prepare.sh` script.
   This generates the `lang` folder and and `global_cmvn` file for the 11800 h / 500 h variant of the training set.
5. Save the `lang` folder and the `global_cmvn` file for later use (e.g., during testing).

##### Variant B: `lang` and `global_cmvn` for the 11800 h / 500 h variant have been generated already

Copy or symlink them into the folder.

#### Step 4

1. Choose a `config.yaml` from `training/configs` corresponding to the chosen model and copy it into the directory.
2. If `alter_epoch.py` is present in the directory with the config, copy it too.

#### Step 5

#####  Variant A: Training from scratch (*baseline* and *norm. -20 dB*)

1. Copy `training/train.sh` into the directory.
2. Set appropriate `NGPUS` (1 for 500 h, 2 for 11800 h) and `NCPUS` at the top of the script.
3. Run the script.

##### Variant B: Finetuning (*Kaldi-like VP*, *proposed VP* and *BVP*)

1. Create a new `train_outs` directory.
2. Copy the `epoch_44.pt` and `epoch_44.yaml` from the `train_outs` directory that was generated during the prior baseline training into the new `train_outs` directory.
3. Run `alter_epoch.py`.
   This modifies the `epoch_44.yaml` to turn on and configure augmentation.
4. Copy `training/train.sh` into the directory.
5. Set appropriate `NGPUS` (1 for 500 h, 2 for 11800 h) and `NCPUS` at the top of the script.
6. Set `INIT_CHECKPOINT=epoch_44.pt` at the top of the script.
7. Run the script.

#### Step 6

The resulting model with be in `train_outs/avg_15.pt`.

## Testing

### Test Data

The following data was used for testing in the paper:

| Dataset    | Subset  | Source Subset  | Selected Recordings                                  |
| ---------- | ------- | -------------- | ---------------------------------------------------- |
| GigaSpeech | —       | test           | in this repository: `test_data_lists/gigaspeech.txt` |
| AMI        | headset | ASR evaluation | `*.Headset-?`                                        |
|            | array   | ASR evaluation | `*.Array1-01`                                        |
| DiPCo      | headset | eval           | `S??_P??`                                            |
|            | array   | eval           | `S??_U??.CH7`                                        |
| CHiME-6    | headset | eval           | `S??_P??`, mono mix                                  |
| MMCSG      | mic. 0  | eval           | all, 1st channel                                     |
|            | mic. 6  | eval           | all, 7th channel                                     |

The selected GigaSpeech recordings have lengths between 100 ms and 60 s.

The ASR evaluation subset of AMI is described here: https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml.

The *Selected Recordings* column uses the standard *glob* syntax for description.

The **transcripts** can be generated using the scripts in `transcription_scripts/test` (see the main block at the bottom of each script and the doc comments of the `generate_transcripts` functions).

### Running the Tests

The following text are instructions for testing the models with the help of the supplied scripts.

There are 3 separate test scripts:

- `test_gigaspeech_normalized.sh` – Tests the power-normalized GigaSpeech test set (for the WER curve from the paper).
- `test_gigaspeech.sh` – Tests the GigaSpeech test set (for the WER table from the paper).
- `test_distant.sh` – Tests the AMI, DiPCo, CHiME-6 and MMCSG datasets (for the WER table from the paper).

The last two are separated since the short length of GigaSpeech samples allows greater parallelization compared to the long recordings from the other datasets.

The training takes place in the following directory structure, whose preparation will be described step-by-step:

- `<current working directory>`
    - `wenet_test` – symlink to `wenet_test`
    - `test_gigaspeech_normalized.sh` – the next 6 files are copied from `testing`
    - `test_gigaspeech.sh`
    - `test_distant.sh`
    - `test_gigaspeech_normalized_config.template.yaml`
    - `test_gigaspeech_config.template.yaml`
    - `test_distant_config.template.yaml`
    - `gigaspeech.list`
    - `distant.list`
    - `lang` – folder generated during 11800 h / 500 h baseline training
    - `global_cmvn` – file generated during 11800 h / 500 h baseline training
    - `train_outs` – symlink or copy of the directory from the training of the tested model
      - `avg_15.pt` – the tested model
    - `results` – an output directory containing the resulting transcripts

#### Step 1

1. Create an empty directory that will serve as the working directory.
2. Add a symlink to `wenet_test` into it.
3. Copy the contents of the `testing` directory into it.

#### Step 2

Prepare the test data as described in [Test Data](#test-data).

Create (or add) the `gigaspeech.list` and `distant.list` files. They have the following format:

```
{"key": "id1", "wav": "path/to/the/first/recording", "txt": ""}
{"key": "id2", "wav": "path/to/the/second/recording", "txt": ""}
```

The `txt` fields can be left empty (these files are not used for WER calculation).
The ids correspond to the ids in the files generated by the scripts in `transcription_scripts/test`.
We do not provide scripts for generating the `.list` files.

The `gigaspeech.list` lists the files from the GigaSpeech test set.
The `distant.list` lists the files from the AMI, DiPCo, CHiME-6 and MMCSG datasets (see the test set table in [Test Data](#test-data) and the outputs of the scripts in `transcription_scripts/test`).

#### Step 3

Symlink the `lang` folder and the `global_cmvn` file obtained during baseline training.

#### Step 4

Symlink a `train_outs` directory from the training of the tested model, containing the `avg_15.pt` file with the final averaged model.

#### Step 5

Configure the variables at the top of the test scripts.

Notably, the `GAINS` variable controls the tested gains.
Set it as `GAINS=(0)` in `test_gigaspeech.sh` and `test_distant.sh` to test only the unaltered recordings.
The other gains in these scripts were used to compute the *oracle power shift* results.

#### Step 6

Run the test scripts.
**1 GPU is required.**
The outputs will be placed in a `results` directory.

#### Evaluation

Run the normalizer in `lib/whisper_normalizer` on the output transcripts (see the scripts in `transription_scripts/test` for examples; we do not provide a script directly for this).

The generation of the reference transcripts is described in the [Test Data](#test-data) section.

We used the [jiwer](https://pypi.org/project/jiwer/) library for WER calculation.
