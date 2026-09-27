# Fire Detection NN

A simple neural network that classifies images as fire or non-fire.

## What it does

1. Loads images from the fire dataset folder
2. Resizes each image to 32×32 and flattens it into a vector
3. Splits the data into train and test (80/20)
4. Trains a small fully connected network
5. Saves the model as `fire_detector_nn.h5`

## Model

- Dense layer, 20 neurons, ReLU
- Dense layer, 8 neurons, ReLU
- Dense layer, 2 neurons, softmax (fire / non-fire)

Optimizer: Adam  
Loss: categorical crossentropy  
Epochs: 10

## Dataset

The image dataset is **not included** in this repo. The script looks for it at:

```
../session17/fire_dataset/
```

You need that folder on your computer before you run the script.

## How to run

Put the dataset next to this project so the path matches the script:

```
../session17/fire_dataset/
```

Then:

```
pip install numpy opencv-python tensorflow scikit-learn
python fire_detection_NN.py
```
