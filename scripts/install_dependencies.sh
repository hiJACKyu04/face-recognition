#!/bin/bash

# Install dependencies script

set -e

echo "Installing dependencies..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Dependencies installed successfully!"

