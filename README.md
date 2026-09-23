<div align="center">

<img width="200" src="https://github.com/rayanrsr/ml-project-template/blob/master/assets/img/icon.png?raw=true">
</img>
<h1>Machine Learning Project Template</h1>

[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Linter: Ruff](https://img.shields.io/badge/Linter-Ruff-brightgreen?style=flat-square)](https://github.com/charliermarsh/ruff)
<br>

Click on [<kbd>Use this template</kbd>](https://github.com/rayanrsr/ml-project-template/generate) to start your own project! or go to the [Documentation](https://rayanrsr.github.io/ml-project-template/) for more information.

<br>
A template for machine learning or deep learning projects.
<br>
<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/palette/macchiato.png" width="400" />
</div>

<br>

## 🧠 Features

- [x] Easy to implement your own model and dataloader through hydra instantiation of datamodules and models
- [x] Configurable hyperparameters with Hydra
- [x] Logging with the solution that fits your needs
- [x] Works on CPU, multi-GPU, and multi-TPUs
- [x] Use bleeding edge UV to manage packages
- [x] pre-commits hooks to validate code style and quality
- [x] Hydra instantiation of models and dataloaders
- [x] torch.compile of models
- [x] Tensors typing validation with TorchTyping
- [x] Dockerized project (Dockerfile, run tests and training through docker, optionally docker-compose)
- [x] Examples of efficient multi-processing using python's pool map
- [x] Examples using polars for faster and more efficient dataframe processing
- [x] Example of mock tests using pytest
- [x] Util scripts to download dataset from kaggle
- [x] Cloud data retrieval using cloudpathlib (launch your training on AWS, GCP, Azure)
- [x] Architecture and example of creating the model serving API through LitServe
- [x] Wiki creation and setup of documentation website with best integrations through Mkdocs

## ⚙️ Steps for Installation

- [ ] Use this repository as a template
- [ ] Clone your repository
- [ ] Run `just install` to install the dependencies (add `--extra cpu`/`--extra gpu` for torch, see below)
- [ ] Add your model which inherits from `LightningModule` in `src/models`
- [ ] Add your dataset which inherits from `Datamodule` in `src/data`
- [ ] Add associated yaml configuration files in `configs/` folder following existing examples
- [ ] Read the commands in the justfile to understand the available commands you can use

`just install` gets you the base dependencies; torch is optional and comes from an extra:

```bash
just install                     # base dependencies only
uv sync --locked --extra cpu     # + torch/torchvision (CPU wheels)
uv sync --locked --extra gpu     # + torch/torchvision (CUDA 11.8 wheels)
```

## 🤠Tips and Tricks

### 🐍 How does the project work?

The `train.py` or `eval.py` script is the entry point of the project. It uses Hydra to instantiate the model (LightningModule), dataloader (DataModule), and trainer using the configuration reconstructed using Hydra. The model is then trained or evaluated using Pytorch Lightning.
The  `serve.py` is used to serve the model through a REST API using LitServe and based on the `configs/serve.yaml` configuration file.

### 👀 Implementing your logic

You don't need to worry about implementing the training loops, the support for different hardwares, reading of
configurations, etc. You need to care about 4 files for each training : your LightningModule (+ its hydra config), your
DataModule (+ its hydra config).

In the LightningModule, you need to implement the following methods:

- `forward method`
- `training_step`
- `validation_step`
- `test_step`

Get inspired by the provided examples in the `src/models` folder.
For the DataModule, you need to implement the following methods:

- `prepare_data`
- `setup`
- `train_dataloader`
- `val_dataloader`
- `test_dataloader`

Get inspired by the provided examples in the `src/data` folder.

Get to know more about Pytorch Lightning's [LightningModule](https://pytorch-lightning.readthedocs.io/en/0.10.0/lightning_module.html) and [DataModule](https://pytorch-lightning.readthedocs.io/en/0.10.0/datamodules.html) in the Pytorch Lightning documentation.
Finally in the associated configs/ folder, you need to implement the yaml configuration files for the model and dataloader.

### 🔍 The power of Hydra

As Hydra is used for configuration, you can easily change the hyperparameters of your model, the dataloader, the trainer, etc. by changing the yaml configuration files in the `configs/` folder. You can also use the `--multirun` option to run multiple experiments with different configurations.

But also, as it used to instantiate the model and dataloader, you can easily change the model, dataloader, or any other component by changing the yaml configuration files or DIRECTLY IN COMMAND LINE. This is especially useful when you want to use different models or dataloaders.

For example, you can run the following command to train a model with a different architecture, changing the dataset
used, and the trainer used:

```bash
uv run src/train.py model=LeNet datamodule=MNISTDataModule trainer=gpu
```

Read more about Hydra in the [official documentation](https://hydra.cc/docs/intro/).

### 💡 Best practices

- Typing your functions and classes with `TorchTyping` for better type checking (in addition to python's typing module)
- Docstring your functions and classes, it is even more important as it is used to generate the documentation with Mkdocs
- Use the `just` commands to run your code, it is easier and faster than writing the full command (and check the justfile for all available commands 😉)
- [Use the pre-commit hooks](https://pre-commit.com/) to ensure your code is formatted correctly and is of good quality
- [UV](https://docs.astral.sh/uv/ ) is powerful (multi-thread, package graph solving, rust backend, etc.) use it as much as you can.
- If you have a lot of data, use Polars for faster and more efficient dataframe processing.
- If you have CPU intensive tasks, use multi-processing with python's pool map, you can find an example in the `src/utils/utils.py` file.

### 📚 Documentation

You have the possibility to generate a documentation website using Mkdocs. It will automatically generate the documentation based on both the markdown files in the `docs/` folder and the docstrings in your code.
To generate and serve the documentation locally:

```bash
just serve-docs # Documentation will be available at http://localhost:8000
```

And to deploy it to Github pages (youn need to enable Pages in your repository configuration and set it to use the
gh-pages branch):

```bash
just deploy-pages # It will create a gh-pages branch and push the documentation to it
```

### 🎓 Github Templates

This repository uses Github templates to help you with issues, pull requests, and discussions. It is a great way to standardize the way your team interacts with the repository. You can customize the templates to fit your needs. They can be find in [.github](.github) folder.

### 🚀 Use this template as your junior's on-boarding process

This template is perfect for your junior's on-boarding process. It has all the best practices and tools to make them productive from day one. It is also a great way to ensure that your team follows the same best practices and tools.
For example you can select as a start a training notebook for any dataset on Kaggle, and ask your junior to
industrialize the notebook into a full-fledged project. It will help them to understand the best practices and tools used in the industry.
After selecting the dataset and notebook, potential steps for the junior can be:

- Implement the DataModule and the LightningModule
- Implement the associated yaml configuration files and use Hydra to instantiate important classes
- Implement the training script
- Implement the evaluation script
- Implement unit tests
- Create a CI/CD pipeline with Github Actions
- Dockerize the project
- Create a justfile with useful commands
- Implement the documentation with Mkdocs
(All of this while following the best practices and tools provided in the template and PEP8)

If any struggle is encountered, the junior can refer to the provided examples in the project.

### 🌳 Tree Explained

```
.
├── assets/commit-template.txt # git commit message template, installed with `just configure-commit-template`
├── configs # configuration files for hydra
│   ├── callbacks # configuration files for callbacks
│   ├── data # configuration files for datamodules
│   ├── debug # configuration files for pytorch lightning debuggers
│   ├── eval.yaml # configuration file for evaluation
│   ├── experiment # configuration files for experiments
│   ├── extras # configuration files for extra components
│   ├── hparams_search # configuration files for hyperparameters search
│   ├── local # configuration files for local training
│   ├── logger # configuration files for loggers (neptune, wandb, etc.)
│   ├── model # configuration files for models (LightningModule)
│   ├── paths # configuration files for paths
│   ├── trainer # configuration files for trainers (cpu, gpu, tpu)
│   └── train.yaml # configuration file for training
├── data # data folder (to store potentially downloaded datasets)
├── build # Dockerfile and docker-compose to build/run the project
├── docs # mkdocs documentation sources
├── justfile # just recipes, the equivalent of a makefile
├── notebooks # notebooks folder
├── pyproject.toml # pyproject.toml file for uv package manager
├── README.md # this file
├── ruff.toml # ruff.toml file for pre-commit
├── scripts # scripts folder
│   └── example_train.sh
├── src # source code folder
│   ├── data # datamodules folder
│   │   ├── components
│   │   └── mnist_datamodule.py
│   ├── eval.py # evaluation entry script
│   ├── models # models folder (LightningModule)
│   │   ├── components # components folder, contains model parts or "nets"
│   ├── train.py # training entry script
│   └── utils # utils folder
│       ├── instantiators.py # instantiators for models and dataloaders
│       ├── logging_utils.py # logger utils
│       ├── pylogger.py # multi-process and multi-gpu safe logging
│       ├── rich_utils.py # rich utils
│       └── utils.py # general utils like multi-processing, etc.
└── tests # tests folder
    └── conftest.py # fixtures for tests
    └── mock_test.py # example of mocking tests

```

## 🤝 Contributing

For more information on how to contribute to this project, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## 🌟 Acknowledgements

This template was heavily inspired by great existing ones, like:

- [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template/)
- [Pytorch Tempest](https://github.com/Erlemar/pytorch_tempest)
- [Yet Another Lightning Hydra Template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
- [Pytorch Style Guide](https://github.com/IgorSusmelj/pytorch-styleguide)
<br>

But with a few opininated changes and improvements, go check them out!
