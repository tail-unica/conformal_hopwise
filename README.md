

<h1 align="center">📊 Conformal hopwise</h1>
<p align="center">
  <b>hopwise extension with conformal risk control.</b>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%7C3.10%7C3.11-green" />
  <img src="https://img.shields.io/github/license/tail-unica/hopwise" />
  <img src="https://img.shields.io/github/repo-size/tail-unica/hopwise">
  <a href="https://github.com/tail-unica/hopwise/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/tail-unica/hopwise"></a>
<a href="https://github.com/tail-unica/hopwise/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/tail-unica/hopwise"></a>
</p>


---

## Overview

**Conformal hopwise** is an advanced extension of the hopwise recommendation framework that supports conformal risk control calibration on top of any recommender system.

By defining a metric that is a proxy for a calibration objective, via a simple evaluation argument in the YAML file, it is possible to create calibrated recommender systems using conformal risk control.

```yaml

# Set Leave-one-out evaluation
eval_args:                     
  split: {'LS': 'valid_calib_and_test'} #{'RS': [0.7, 0.1, 0.1, 0.1]}   # (dict) The splitting strategy ranging in ['RS','LS'].
  group_by: user                
  order: TO                     # (str) The ordering strategy ranging in ['RO', 'TO'].
  mode: {"valid": "full", "calib": "full", "test": "full"}


# Calibration Arguments
calibration:
    calibrator: ConformalRiskControl # name of the conformal calibrator to use for calibration (e.g., ConformalRiskControl, etc) present in hopwise.model.conformal_calibration.py
    alpha: 0.60 # 1 - confidence level (e.g., 0.60 in case of FNR means we want to ensure that is guaranteed that in expectation across users, the recall is 40%)
    maximum_metric_score: 1.0 # maximum possible value of the metric (e.g., 1.0 for novelty, 1 for recall, etc..)
    n_lambdas: 3000 # the number of decision thresholds to consider when calibrating the model (e.g., 50000 means we will consider 50000 different lambda thresholds)
    loss: FNR # the class name of the loss to use for calibration (e.g., FNR, FPR, etc..) present in hopwise.evaluator.conformal.metrics.py that is a proxy for the guarantees we are interested in achieving

    # EXPECTATION CHECK
    prove_expectation: False
    n_calibration_users: 0.4 # the proportion of users to use for calibration (the rest will be used for testing)
    n_trials: 500 # to verify the expectation of the conformal risk control
```

**Requirements**
1. The chosen model should implement a `full_sort_predict()` function. For example, hopwise.model.general_recommender.bpr already has it.
2. The evaluation should be set to leave-one-out, as shown above.
3. The calibration arguments should be set.

---

## ⚡ Installation

To install the project, you need to use `uv`. Follow the steps below to set up the environment and install the necessary dependencies.

## 🔹 Prerequisites
- ✅ Python **3.9**, **3.10**, or **3.11**
- ✅ [`uv`](https://github.com/astral-sh/uv) package manager

---

### 🔹 Steps (from PyPI or from Source)


1️⃣ **Install **uv** and create a virtual environment.**<br>

We suggest installing **uv** as a [standalone application](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) instead of using pip to avoid issues and let **uv** create a dedicated virtual environment.<br>
Once installed, create the virtual environment

```sh
uv venv --python PYTHON_VERSION --prompt hopwise
```
`PYTHON_VERSION` must be one of 3.9, 3.10, 3.11, while `--prompt hopwise` customizes the virtual environment name that appears on the shell.

2️⃣ **Install via PyPI**

```sh
uv pip install hopwise
```

Some models require extra dependencies.
In particular, language models for KG path reasoning require extra dependencies to be installed.
You can install them by specifying the extra `pathlm` in the command line as follows:
```sh
uv pip install hopwise[pathlm]
```

Other models can be installed with a similar process. For instance, to install NNCF:
```sh
uv pip install hopwise[nncf]
```

Please check the [PyPI page](https://pypi.org/project/hopwise/) for the complete list of extra dependencies and the [documentation](https://hopwise.readthedocs.io/en/latest/installation.html#extra-dependencies) for more details on how to install hopwise with specific dependencies.

**🎉 Done 🎉**

---

2️⃣ **Install from source: Clone the repository**
```sh
git clone https://github.com/tail-unica/hopwise.git
cd hopwise
```


3️⃣ Install project dependencies

*📌 make sure to have uv updated to the latest version*

```sh
uv sync
```

> 📢 **Windows:** For proper DGL installation, please follow the [official DGL installation guide](https://www.dgl.ai/pages/start.html). Windows builds may encounter DLL linking issues with standard installation methods. Pre-built packages from the official source are recommended. Otherwise, using the Windows Subsystem for Linux (WSL) might be feasible as a solution.

**🎉 Done 🎉**

## 🚀 Usage
In any chosen setup, a .yaml file must be created containing the configuration to be used. An example:
```yaml
gpu_id: 0
topk: [10,20,50,...]
data_path: *your_datasets_folder*
metrics: ['NDCG', 'MRR', 'Hit', 'Precision', 'Recall',...]
valid_metric: ndcg@10
eval_batch_size: 1
epochs: 1
eval_step: 1
....

# evaluation config
... see above


# calibration config
... see above

```

### 📍 Training

<p align="center">
    <a href="#readme">
        <img alt="traintest" src="https://github.com/tail-unica/hopwise/blob/main/assets/trainpgprclip.gif">
    </a>
</p>


Run the project with the following command:
```sh
hopwise train \
    --model MODEL \
    --dataset DATASET \
    --config_files CONF_FILE_1.yaml CONF_FILE_2.yaml
```

Override config parameters directly from the CLI using =:
```sh
hopwise train --epochs=20
```

### 📍 Evaluating from Checkpoint
<p align="center">
    <a href="#readme">
        <img alt="pgprevaltest" src="https://github.com/tail-unica/hopwise/blob/main/assets/pgprevaluation.gif">
    </a>
</p>

```sh
hopwise evaluate --dataset DATASET --model MODEL \
--config-files CONFIG_FILES --checkpoint CHECKPOINT.pth
```

### 📍 Hyperparameters Tuning

In addition to the configuration file, a params file with the extension *.hyper* the range of hyperparameters to be tested must also be set in this configuration

```yaml
learning_rate uniform 0.0001, 0.1
embedding_size choice [64, 100, 200]
```

<p align="center">
    <a href="#readme">
        <img alt="hypertuningtest" src="https://github.com/tail-unica/hopwise/blob/main/assets/hypertuningbpr.gif">
    </a>
</p>

```sh
hopwise tune \
    --params-file hopwise/properties/hyper/PARAMS_FILE.hyper \
    --config-files CONFIG_FILE.yaml \
    --study-name STUDY_NAME
```

## ℹ️ Contributing
Please let us know if you encounter a bug or have any suggestions by filing an issue.

We welcome all contributions from bug fixes to new features and extensions. 🚀

We expect all contributions discussed in the issue tracker and going through PRs. 📌

## 📜 Cite
If you find **hopwise** useful for your research or development, please cite with:

```bibtex
the paper is under revision
```

## The Team

[Ludovico Boratto](https://www.ludovicoboratto.com/), [Gianni Fenu](https://web.unica.it/unica/it/ateneo_s07_ss01.page?contentId=SHD30371), [Mirko Marras](https://www.mirkomarras.com/), [Giacomo Medda](https://jackmedda.github.io/), [Alessandro Soccol](https://alessandrosocc.github.io)




## License
This project is licensed under the MIT License. See the LICENSE file for details.

