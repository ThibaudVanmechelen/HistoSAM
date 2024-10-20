# SAMSAM

This repo aims to provide a simple way to train Segment Anything Model on digital pathology data.

This repository contains code to train Segment Anything Model on digital pathology image. Additionally, it offers scripts to scrap and preprocess data from cytomine platform.

## TODO
- [x] Implement training pipeline
- [x] Implement train/test split
- [ ] Implement interactive prompting tool for the user

## Installation

### Clone the repository

```bash
git clone https://github.com/NoeGille/SAMSAM.git
```

### Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```

## Project configuration

For easier configuration of the project, we provide a `config.toml` file in the root of the project. This file contains all the necessary configuration for the project.
For most executables, you can specify your own toml file which will be used instead of the default one (It has to follow the same structure).
```bash
python samsam.py --config path/to/your/config.toml
```

Furthermore, if you want to use scrapers script on the Cytomine platform for your own datasets, you need to create a keys.toml file in the root of the project. This file should contain the following information:
```toml
title = "Cytomine API keys"
host = "[cytomine link]"
public_key = "[your public]"
private_key = "[your private key]"
```
Do not share this file with anyone.
Please refers to the Cytomine API Documentation for more information. https://doc.uliege.cytomine.org/

## Usage

It is recommended to use the SAMDataset class from `dataset_processing/dataset.py` to load the data and give it to the model. The class handle the processing of the data including automatic prompting.

SAM is used throught a child class named TrainableSam. This class simplifies the use of SAM in a training pipeline without changing the inner functioning of SAM.

### Evaluation

As it is not mandatory to fine-tune the model, you can directly evaluate the model on your dataset. You can refer to evaluate.py in this case

### Training

Training is done through the TrainableSam class. You can refer to train.py in this case. You can either train the whole model (including the image encoder) or only the prompt encoder and decoder. In the latter case, use `save_img_embeddings.py` to save the image embeddings then `use_img_embeddings option` in `config.toml` to skip the image encoder part for faster training.

# Acknoledgement

Thanks to Pierre GEURTS and Raphael MAREE for their help on this project.

Moreover, Please consider to checkout the original Segment Anything Model paper for more details on the model architecture and training pipeline :
```bibtex
@misc{kirillov2023segment,
      title={Segment Anything}, 
      author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
      year={2023},
      eprint={2304.02643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.02643}, 
}
```