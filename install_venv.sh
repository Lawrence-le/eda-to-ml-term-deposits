#!/bin/bash

echo "==========================================="
echo "Checking and Installing Virtual Environment and Jupyter Kernel"
echo "==========================================="

# Set Python command based on the OS
if [[ "$OSTYPE" == "msys" ]]; then
  PYTHON_CMD="python"  # For Windows
else
  PYTHON_CMD="python3"  #For macOS/Linux
fi

# Check if virtual environment exists; if not, create it
if [ ! -d "venv" ]; then
  echo "Virtual environment not found. Creating a new one..."
  $PYTHON_CMD -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]]; then
  # For Windows (Git Bash)
  source venv/Scripts/activate
else
  # For macOS/Linux
  source venv/bin/activate
fi

# Install ipykernel in the virtual environment if it's not already installed
echo "Installing ipykernel in the virtual environment..."
pip install ipykernel

# Add the virtual environment as a Jupyter kernel
echo "Adding virtual environment as Jupyter kernel..."
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

echo "==========================================="
echo "Virtual Environment and Jupyter Kernel Setup Complete"
echo "==========================================="
