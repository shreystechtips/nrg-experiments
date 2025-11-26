# Lightweight PhotonVision Python implementation
This project aims to replicate basic PhotonVision behavior using Python. The aim is to increase frame processing performance over the existing Java implementation and give more control for us to be able to customize the software to our liking. 
Frankly, PhotonVision is a massive black box, which makes it really hard to improve its accuracy and performance. This project is a way to me to understand how it works at a deeper level so we can improve it in the future. 


## Usage
- **Install**: We use `uv` to manage the Python environment. Please ensure that `uv` is already installed ([guide from official website](https://docs.astral.sh/uv/getting-started/installation/)). tldr: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Execute**: Simply run `uv run server.py` from this folder. For arm64-linux targets, the first startup will be slow because wheels for `pupil-apriltags` are not published by default. 
- **Auto execute**: This is still a work in progress. I hope to set up instructions on how to ensure that this server starts up on boot and auto recreates if the process dies.


## Credits
### AI warning
The initial implementation of this project was largely wireframed in Grok. Since then, I have rewritten ~90% of it to improve code readability and structure, processing performance and overhead, and the outputs it produces. The HTML UI largely remains AI coded.
### PhotonLib adaptation
Swaths of code were adapted from [`photonlibpy`](https://github.com/PhotonVision/photonvision/tree/main/photon-lib/py/photonlibpy) as they proved useful in helping to understand the system and also pack data in ways compatible with the PhotonVision NetworkTables API.
