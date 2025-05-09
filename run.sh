#!/bin/bash

echo "==========================================="
echo "Starting Machine Learning Pipeline..."
echo "==========================================="

# Function to display the loading spinner
# https://stackoverflow.com/questions/12498304/using-bash-to-display-a-progress-indicator-spinner
spin()
{
    local pid=$!
    local delay=0.75
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Set Python command based on the OS
if [[ "$OSTYPE" == "msys" ]]; then
  PYTHON_CMD="python"  # For Windows
else
  PYTHON_CMD="python3"  #For macOS/Linux or GitHub Actions
fi

# Check if virtual environment exists; if not, create it
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  $PYTHON_CMD -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]]; then
  # For Git Bash on Windows
  source venv/Scripts/activate
else
  # For macOS/Linux or GitHub Actions
  source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Run model script
echo "Running model script..."
echo "!!!! Report for Models may only appear after Graph is CLOSED !!!!"
$PYTHON_CMD src/model.py & spin

echo "==========================================="
echo "Pipeline Complete!"
echo "==========================================="
