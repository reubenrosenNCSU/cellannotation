# Automatic Cell Annotation Tool

[![YouTube Demonstration](https://img.shields.io/badge/YouTube-Demonstration-red)](https://youtu.be/IhLrQrVeXEQ)

A machine learning-based tool for automated detection and annotation of **Spiral Ganglion Neurons (SGN)** and **Mosaic Analysis with Double Markers (MADM)** cells in microscope images. Built on the `keras-retinanet` framework, this tool enables biologists to deploy and refine object detection models without requiring programming expertise.

![Screenshot](screenshots/image.png)

## Features

- **Automated Detection**: Pre-trained models for SGN and MADM cell detection.
- **Real-Time Analysis**: Dynamic cell counting during image processing.
- **Annotation Editing**: Add, modify, or remove annotations post-detection.
- **Model Customization**: 
  - Fine-tune models using custom datasets or prior annotations.
  - Retrain models for improved accuracy.
- **Image Preprocessing**: Adjust brightness, contrast, and zoom; set brightness/contrast thresholds.
- **Cross-Platform Accessibility**: Hosted on a static IP for network-wide access.

---

## Installation

### Prerequisites
- **Anaconda**: Install from [Anaconda Documentation](https://docs.anaconda.com/anaconda/install/index.html).
- **Linux Environment**: Tested on Ubuntu 20.04 LTS.

### Steps

1. **Clone and Configure Repositories**:
   ```bash
   git clone https://github.com/fizyr/keras-retinanet.git
   mv keras-retinanet keras_retinanet
   ```
**Replace Core Files**:

Overwrite the following files in keras_retinanet/utils/ with those provided in this repository:

```
image.py
```
```
colors.py
```
```
gpu.py
```
**Modify training script**:

In keras_retinanet/keras_retinanet/bin/train.py, set steps = None (default: 10000).

Set Up Conda Environment:


```
conda env create -f environment.yml
conda activate environment
pip install -r requirements.txt
```
**Install keras-retinanet in Editable Mode:**


```
cd keras_retinanet
pip install cython numpy
python setup.py build_ext --inplace
pip install --use-pep517 -e .
```

**Verify Installation**:

```
python
import keras_retinanet
print(keras_retinanet.__file__)  # Ensure path points to the local repository.
```

## Deployment
**Directory Structure**

Create the following directories if they do not exist:

```
uploads/
images/
input/
output/output_csv/
ft_upload/
saved_annotations/
saved_data/
converted/         # Clear manually to avoid PNG accumulation
finaloutput/
snapshots/         # Critical for model weights
```

**Pre-Trained Weights**

Download and place these files in /snapshots:

[SGN_Rene.h5](https://drive.google.com/file/d/10JCk6W6pC7nVWfHJ7Ew6xvyWLEeKxbV2/view?usp=sharing)
[combine.h5](https://drive.google.com/file/d/1ADUyTbD1wxKvsMnuvF0YZr5K9Wn5iwk3/view?usp=sharing)

Launch Application
Run the server:

```
python app.py
```
Access the tool via browser:

```
http://127.0.0.1:5000/static/index.html
```
For network access, replace 127.0.0.1 with the host machine's IP.

**Directory Structure Example**

![Screenshot](https://camo.githubusercontent.com/804f51b9960a47677c5fbb0f0a504e35b0f85b6118ac9c7ef096837383f689f4/68747470733a2f2f692e6962622e636f2f33794d66533079462f696d6167652e706e67)

## Acknowledgements

 - [COMBINe: Cell detectiOn in Mouse BraIN](https://github.com/yccc12/COMBINe/tree/main)
 - [keras-retinanet](https://github.com/fizyr/keras-retinanet)
