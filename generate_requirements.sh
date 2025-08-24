#!/bin/bash

python -m piptools compile pyproject.toml --extra dev --output-file requirements-dev.txt
python -m piptools compile pyproject.toml --output-file requirements.txt
