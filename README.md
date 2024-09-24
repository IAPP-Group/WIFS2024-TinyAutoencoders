# Tiny_GM_Detectors

This repository contains the implementation of the paper **"Tiny Autoencoders Are Effective Few-Shot Generative Model Detectors"**, accepted at WIFS 2024.

![Network Architecture](images/architecture.png)

## Prerequisites

- Python 3.x
- PyTorch
- [Flickr30k Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

You will need to modify the script paths to point to your local dataset directories.

## Setup

1. Clone the repository:

    ```bash
    git clone /Tiny_GM_Detectors.git
    cd Tiny_GM_Detectors
    ```

2. Install the necessary Python packages by running:

    ```bash
    pip install torch torchvision
    ```

## Scripts

### 1. Pretrain the Autoencoder

The first step is to pretrain the real autoencoder using the `Flickr30k` dataset. You need to specify the path to the dataset in the script.

    ```bash
    python pretrain_real_ae.py
    ```


### 2. Train Tiny Detectors

Once the autoencoder is pretrained, you can train the tiny encoder detectors for the generative models.

    ```bash
    python train_tiny_detectors.py
    ```


### 3. Compute Feature Vectors

Compute the feature vectors by extracting the reconstruction errors from the trained encoders.

    ```
    python compute_features.py
    ```

This generates the feature vectors for model evaluation.

## Citation

If you use this repository in your research, please cite the following paper:

**"Tiny Autoencoders Are Effective Few-Shot Generative Model Detectors"**  
*Authors: Luca Bindini, Giulia Bertazzini, Daniele Baracchi, Dasara Shullani, Paolo Frasconi, Alessandro Piva*  
Accepted at WIFS 2024

    ```bibtex
    @inproceedings{bindini2024,
      title={Tiny Autoencoders Are Effective Few-Shot Generative Model Detectors},
      author={Luca Bindini and Giulia Bertazzini and Daniele Baracchi and Dasara Shullani and Paolo Frasconi and Alessandro Piva},
      booktitle={Proceedings of the IEEE International Workshop on Information Forensics and Security (WIFS)},
      year={2024}
    }
    ```
