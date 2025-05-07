# !/bin/bash

echo "Starting ML Pipeline..."

# Activate the virtual environment
source venv/bin/activate
# source venv/Scripts/activate  # For Windows via GitBash

# Install dependencies
pip install -r requirements.txt

# Run Python Scripts
# python3 src/data_loader.py
# python3 src/preprocess.py
# python3 src/train.py
# python3 src/evaluate.py
python3 src/hello.py