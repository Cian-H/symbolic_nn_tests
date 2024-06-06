# Overview

This is a testing sandbox for developing various methods of injecting symbolic knowledge into neural networks. This symbolic NN approach is intended to facilitate the training of models for physics based problems where process complexity and data availability are generally significant hindrances to the application of NNs. By injecting known, symbolic human knowledge we aim to guide the model towards the correct embedding space more quickly than could be achieved through pure statistical loss.

# Experiments

## Experiment 1 - Semantically weighted loss

- Simple semantic loss based on intuitively noticeable properties of numeric characters
- Makes use of manually made reward matrices to weight rewards depending on how "close" the model was
- Very rudimentary example of semantic loss
- Appeared to work perfectly

## Experiment 2 - Semantic loss function

- Makes use of known physics equations that partially describe the problem to guide the model
    - Reduces the need for the model to learn known physics, allowing it to focus on learning the unknown physics
    - Should accelerate training
    - Allows for easier model minimisation
- Dataset selection will be important here
    - Needs to be a familiar area to the experimenter
    - Needs to be openly available
    - Needs to describe a problem complex enough that we wouldn't just be fitting on a known equation (e.g: if we were to try Hooks law we would just fit to the equation which would be pointless)
- Possible candidate datasets:
    - [Molecular Properties](https://www.kaggle.com/datasets/burakhmmtgl/predict-molecular-properties)
    - [Nuclear Binding Energy](https://www.kaggle.com/datasets/iitm21f1003401/nuclear-binding-energy)
    - [Body Fat Prediction](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)
