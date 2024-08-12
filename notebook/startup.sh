#!/bin/bash

# Check if python3 or python is available
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

# Create a virtual environment
$PYTHON -m venv venv

# Check if venv was created successfully
if [ $? -ne 0 ]; then
    echo "Failed to create a virtual environment. Make sure that venv is installed."
    exit 1
fi

# Determine the shell and activate the virtual environment accordingly
case "$SHELL" in
*/bash)
    source venv/bin/activate
    ;;
*/zsh)
    source venv/bin/activate
    ;;
*/fish)
    source venv/bin/activate.fish
    ;;
*)
    echo "Unsupported shell. Please activate the virtual environment manually."
    exit 1
    ;;
esac

# Windows specific activation (for Git Bash or other bash-like environments)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
fi

# Install the required packages
pip install -r requirements.txt

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo "Setup complete. Virtual environment is ready and dependencies are installed."
else
    echo "There was an error installing the dependencies."
    exit 1
fi

# Install the virtual environment as a Jupyter kernel
$PYTHON -m ipykernel install --user --name=venv --display-name "Python (venv)"
echo "Jupyter kernel 'Python (venv)' has been installed."

echo "Setup is complete. To use the 'Python (venv)' kernel in Jupyter Notebook, start Jupyter and select the kernel from the 'Kernel > Change Kernel' menu."
