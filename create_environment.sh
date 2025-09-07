#!/bin/bash

echo "Setting up ML project environment..."

# Function to check if conda is installed
check_conda() {
    if command -v conda &> /dev/null; then
        echo "Conda is already installed."
        return 0
    else
        echo "Conda is not installed."
        return 1
    fi
}

# Function to install Anaconda
install_anaconda() {
    echo "Installing Anaconda..."
    
    # Download Anaconda installer
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O anaconda.sh
    
    # Make installer executable
    chmod +x anaconda.sh
    
    # Run installer silently
    bash anaconda.sh -b -p $HOME/anaconda3
    
    # Add conda to PATH
    echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
    
    # Initialize conda
    source ~/.bashrc
    $HOME/anaconda3/bin/conda init bash
    
    # Clean up
    rm anaconda.sh
    
    echo "Anaconda installation completed!"
}

# Check if conda is installed, install if not
if ! check_conda; then
    install_anaconda
    # Source bashrc to get conda in current session
    export PATH="$HOME/anaconda3/bin:$PATH"
fi

# Create conda environment
echo "Creating conda environment 'ml-project'..."
conda create -n ml-project python=3.9 -y

# Activate environment
echo "Activating ml-project environment..."
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate ml-project

# Install packages from dev_requirements.txt
echo "Installing packages from dev_requirements.txt..."
pip install -r dev_requirements.txt

# Install Jupyter kernel for this environment
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name ml-project --display-name "ML Project"

echo "Environment setup completed successfully!"
echo "To activate this environment in the future, run:"
echo "conda activate ml-project"
