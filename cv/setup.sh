#!/bin/bash

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: This script is intended for Linux only." >&2
    exit 1
fi


SCRIPT_DIR=$(dirname "$0")
cd SCRIPT_DIR

# Install the Python package manager.
if ! command -v uv &> /dev/null
then
    echo "uv not found, installing..."
    # Download and run the official installation script
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # The installer puts uv in ~/.cargo/bin by default.
    # This block ensures the directory is added to the PATH for the current session, if needed.
    export PATH="$HOME/.cargo/bin:$PATH"

    echo "uv installed successfully."
else
    echo "uv is already installed."
fi

# Install the environment data
uv sync

# Put the service in the right place.
sudo cp photonvision.service  /lib/systemd/system/photonvision.service
# Force the service to be picked up.
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable photonvision.service
sudo systemctl start photonvision.service
