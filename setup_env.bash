#!/bin/bash

HELP_MESSAGE="Usage: ./build.sh [-r, --rebuild]
Build the '${IMAGE_NAME}' image.
Options:
  -r, --rebuild            Remove the previous venv completely before rebuilding
  -h, --help               Show this help message."

# Parse input flags
BUILD_FLAGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
  -r | --rebuild)
    rebuild=true
    shift 1
    ;;
  -h | --help)
    echo "${HELP_MESSAGE}"
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    exit 1
    ;;
  esac
done

# Check if we need to remove past files and caches
if [ "$rebuild"=true ] ; then
    echo "Removing previous venv..."
    rm -rf .venv
    echo "Done"
fi

echo "Creating venv..."
python3.9 -m venv .venv          # Create venv (use any python version you like >=3.9)
echo "Done"

echo "Activating venv..."
source .venv/bin/activate        # Activate venv
echo "Done"

pip install -r requirements.txt  # Install package requirements
python3.9 -m build               # Build package
python3.9 -m pip install -e .    # Install package. -e for editable, developer mode. 