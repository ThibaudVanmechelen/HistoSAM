# HistoSAM: An Approach Using Segment Anything Models for Histopathology

## Description
This repository was developed as part of my master's thesis at [Uliège](https://www.uliege.be/cms/c_8699436/en/uliege), which focused on the use of the Segment Anything Model (SAM) and Segment Anything Model 2 (SAM2) for histopathological data. The goal was to experiment with these models and explore how they could be fine-tuned to improve performance. The best-performing model would then be deployed on the [Cytomine](https://uliege.cytomine.org/) platform.

In this repository, you'll find code to fine-tune, test, and experiment with both Segment Anything models. You'll also find the HistoSAM model, which combines the Segment Anything Model with several domain-specific histopathology encoders. The concept behind HistoSAM is generalizable to any encoder, as long as it is a ViT (Vision Transformer) architecture.

# Requirements
Python 3.12+

# Installation
All required libraries can be installed using the following command:
```bash
pip install -r requirements.txt
```

# Usage
Since fine-tuning the Segment Anything Models is computationally intensive, no direct script is provided to run the code out-of-the-box.

Additionally, access to the Cytomine server used during the project is not publicly available. If you'd like to reuse or adapt parts of this codebase, please refer to the documentation. Several example configuration files are provided in the configs folder.

# Configuration
To use the Cytomine API, you need to create a keys.toml file with the following format:

## Keys.toml format

```bash
title = "Cytomine API keys"
host = "your_host"
public_key = "your_public_key"
private_key = "your_private_key"
```

# License
MIT

# Credits
This codebase was partially based on work developed during an internship, available [here](https://github.com/NoeGille/SAMSAM).

# Contact
If you have questions about the code or other aspects of this repository, feel free to open a new discussion on GitHub. However, please note that I cannot guarantee a response.