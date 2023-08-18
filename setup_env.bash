#!/bin/bash

python3.9 -m venv .venv          # Create venv (use any python version you like >=3.9)
source .venv/bin/activate        # Activate venv
pip install -r requirements.txt  # Install package requirements
python -m build                  # Build package
pip install -e .                 # Install package. -e for editable, developer mode. 