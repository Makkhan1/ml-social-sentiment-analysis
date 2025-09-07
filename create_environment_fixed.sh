#!/bin/bash

echo "Setting up ML project environment with Miniconda..."

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

# Function to install Miniconda (lighter than Anaconda)
install_miniconda() {
    echo "Installing Miniconda..."
    
    # Download Miniconda installer (much smaller and more reliable)
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Make installer executable
    chmod +x miniconda.sh
    
    # Run installer silently with no multiprocessing issues
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    
    # Initialize conda
    source ~/.bashrc
    $HOME/miniconda3/bin/conda init bash
    
    # Clean up
    rm miniconda.sh
    
    echo "Miniconda installation completed!"
}

# Check if conda is installed, install if not
if ! check_conda; then
    install_miniconda
    # Source bashrc to get conda in current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# Update conda to latest version
echo "Updating conda..."
conda update conda -y

# Create conda environment
echo "Creating conda environment 'ml-project'..."
conda create -n ml-project python=3.9 -y

# Activate environment
echo "Activating ml-project environment..."
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ml-project

# Install packages from dev_requirements.txt
echo "Installing packages from dev_requirements.txt..."
while IFS= read -r package; do
    echo "Installing $package..."
    pip install "$package"
done < dev_requirements.txt

# Install additional useful packages
echo "Installing additional packages..."
conda install jupyter notebook ipykernel -y

# Install Jupyter kernel for this environment
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name ml-project --display-name "ML Project"

echo "Environment setup completed successfully!"
echo ""
echo "To activate this environment in the future, run:"
echo "conda activate ml-project"
echo ""
echo "To deactivate, run:"
echo "conda deactivate"
