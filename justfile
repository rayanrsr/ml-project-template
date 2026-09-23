# Variables used by several recipes.
# `just install PYTHON_VERSION=3.12` (or an exported PYTHON_VERSION) overrides the default.
PYTHON_VERSION := "3.12"
DOCKER_RUN_FLAGS := '--privileged --network=host --ulimit nofile=65536:65536'
DOCKER_RUN_FLAGS_GPU := '{{DOCKER_RUN_FLAGS}} --gpus all'

# `just` exports variables to the recipes, so the training/eval scripts find the project root
# and can import `src.*` without any manual PYTHONPATH/PROJECT_ROOT export.
export PROJECT_ROOT := justfile_directory()
export PYTHONPATH := justfile_directory()

[group('setup')]
init:
	uv init -p {{PYTHON_VERSION}}

[group('setup')]
install: configure-commit-template
	uv sync --locked -p {{PYTHON_VERSION}}

# Configure git commit template, this will help to write better commit messages
[group('setup')]
configure-commit-template:
	git config --global commit.template {{justfile_directory()}}/assets/commit-template.txt

# Pre-commit hooks are useful to run some checks (usually linters) before committing the code to the repository
# It will help to keep the code clean and consistent, this commands sets up the pre-commit hooks
[group('setup')]
configure-pre-commit:
	uvx pre-commit install

# Run pre-commit hooks on all files, it will run the checks on all files in the repository
[group('ci')]
format:
	uvx pre-commit run --all-files

# Run pytest to test the code unit tests under the 'tests/' directory
[group('ci')]
test:
	uv run --no-sync pytest

# Use uv to run the train script while passing the arguments from the command line
[group('code')]
train:
	uv run --no-sync src/train.py {{ARGS}}

# Use uv to run the evaluate script while passing the arguments from the command line
[group('code')]
evaluate:
	uv run --no-sync src/eval.py {{ARGS}}

# Use uv to run the serve script while passing the arguments from the command line
[group('code')]
serve:
	uv run --no-sync src/serve.py {{ARGS}}

# Build the Docker image with the base dependencies
[group('docker')]
build-docker:
	docker build --target lightning-base -t lightning-base:latest -f build/Dockerfile .

# Build the Docker image and jump into the container to test a fresh environment (CPU)
[group('docker')]
dev-container-cpu: build-docker
	docker run {{DOCKER_RUN_FLAGS}} -v {{justfile_directory()}}:/app -it lightning-base:latest /bin/bash

# Build the Docker image and jump into the container to test a fresh environment (GPU)
[group('docker')]
dev-container-gpu: build-docker
	docker run {{DOCKER_RUN_FLAGS_GPU}} -v {{justfile_directory()}}:/app -it lightning-base:latest /bin/bash

# Run the train script using the Docker image
[group('docker')]
train-docker: build-docker
	docker run {{DOCKER_RUN_FLAGS}} --user root -v {{justfile_directory()}}:/app -w /app lightning-base:latest /bin/bash -i -c "uv run /app/src/train.py {{ARGS}}"

# Run the evaluate script using the Docker image
[group('docker')]
evaluate-docker: build-docker
	docker run {{DOCKER_RUN_FLAGS}} --user root -v {{justfile_directory()}}:/app -w /app lightning-base:latest /bin/bash -i -c "uv run /app/src/eval.py {{ARGS}}"

# This build the documentation based on current code 'src/' and 'docs/' directories and deploy it to the gh-pages branch
# in your GitHub repository (you then need to setup the GitHub Pages to use the gh-pages branch)
[group('docs')]
deploy-pages:
	uv run --extra docs mkdocs build -f docs/mkdocs.yml && uv run --extra docs mkdocs gh-deploy -f docs/mkdocs.yml

# This is to run the documentation locally to see how it looks
[group('docs')]
serve-docs:
	uv run --extra docs mkdocs build -f docs/mkdocs.yml && uv run --extra docs mkdocs serve -f docs/mkdocs.yml
