#!/bin/bash
echo "HELLO"
venv_name="venv"
requirements_file="./requirements.txt"  # Replace with the actual path to your requirements.txt

# Create virtual environment
python -m venv $venv_name

# Activate virtual environment
source ./$venv_name/bin/activate  # On macOS/Linux
#.\$venv_name\Scripts\activate.bat

# Install dependencies from requirements.txt
pip install -r $requirements_file
echo "Installed files"
# Deactivate virtual environment
deactivate