# LSTM-Autoencoder for Outlier Detection

## Background

### Description

In this project, an LSTM Autoencoder is trained and applied in the context of outlier detection. The dataset used for this task is the [Ozone Level Detection dataset](https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection). The task is to use weather and enviromental data to predict an extreme weather event.

### Why anomoly detection, specifically an LSTM Autoencoder?

Due to the limited number of extreme weather events, anomoly detection in this context serves as the apropriate tool to use. A model can be constructed which learns from the normal weather events, and applied with the aim of recognising extreme events as anomolies. 


The core idea behind an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) is to learn a representation can be used to reconstruct the original input sequence. In the context of our problem, detecting extreme weather events, we want to train a model to reconstruct normal weather events which ultimately leads to a learnt model able to detect ‘normality’ as this is all the model has seen. 

In the cases of extreme weather events, the model has not ‘seen’ these types of sequences before, and ultimately has trouble reconstructing the sequence; this leads to a high reconstruction error. The reconstruction error then acts as a metric whereby we can set some threshold, i.e. if reconstruction error is > some_threshold must be an extreme weather event.

There are ways to set the threshold algorithmically, for now, I have used manual inspection (see below for further details).

Why an [LSTM-autoencoder](https://blog.keras.io/building-autoencoders-in-keras.html)? Due to weather events exhibiting a temporal nature, an LSTM can make use of the long-range dependencies inherit in the dataset. This approach assumes there is a temporal nature / ramp-up before an extreme weather event.

<img src="/figures/lstm-ae.png"> 

### What about other approaches?

I did ponder other approaches, rather than take the non-parametric discriminative route, one could use a generative approach to model the underlying distribution such as a [variational autoencoder LSTM](https://towardsdatascience.com/time-series-generation-with-vae-lstm-5a6426365a1c) which would additionally provide certainty estimates. Besides this, a classical approach could also be to construct a distribution of normal weather events and use this as a reference when testing for extreme weather events. 

## Running this repo

The specific Python version is 3.9 using PyTorch CUDA 11. Packages can be installed from the included requirements.txt. To the run the code, execute main.py with desired parameters. 


## Technical Details

### Process

This section describes the end-to-end logic within this pipeline.

1. Start by loading the ozone data, append the labels and perform some basic cleaning.
1. Remove features where the absolute correlation > some threshold.
1. Split the data into, train, valid, test and anomoly.
1. Impute missing values by [Multiple Imputation by Chained Equations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/)
1. Normalise and standardise the dataset.
1. Convert the data to PyTorch tensors.
1. Train/Load an LSTM-AE with/without dropout layer.
1. Perform the usual evaluation etc.

### Model Details

Standard parameters have been set for this model, the representation size is N=32. The model uses two LSTM layers in the encoder as well as decoder. The model could do with some hyperparameter exploration to choose the most appropriate parameters. 

## Evaluation

This section discusses the evalaution of the chosen approach. For reference, I did not perform any hyperparameter exploration such as a [grid search](https://scikit-learn.org/stable/modules/grid_search.html). I simply tinkered with the architecture manually.

### Loss curves

The below image depicts the training and validation loss curves. The model tends to generalise around 75 epochs.

<img src="/figures/model.png"> 


### Training Reconstruction Loss

The below image depicts the reconstruction loss for the training set. Based on this distribution we can set a threshold at the peak of the distribution around 0.07.

<img src="/figures/train_reconstruction_loss.png"> 


### Anomoly Reconstruction Loss

The below image depicts the reconstruction loss for the anomoly set. As seen there is overlap between the threshold set and the ability to reconstruct the anomoly dataset. 

<img src="/figures/anomoly_reconstruction_loss.png"> 


### Performance

Using both the testing and anomoly set, I am ebale to compute the [F1 metric](https://en.wikipedia.org/wiki/F-score) (micro) which takes into account FP/FN/TP/TN etc. The perfect score is 0.


Encoder Layers | Decoder Layers | Hidden Units | F1-Micro | 
--- | --- | --- | --- | 
2 | 2 | 64 | 0.61  
2 | 2 | 32 | 0.70  
2* | 2* | 16* | 0.75*
1 | 1 | 64 | 0.58
1 | 1 | 32 | 0.64
1 | 1 | 16 | 0.61


The most accurate method was a model with 2 layers in the encoder, 2 layers in the decoder and 16 hidden units. The network is able to predict all extreme weather events, however, due to false positives the F1 micro is 75%.

## Discussion

Given more time, one would also explore the following:

1. Exhuastive Hyperparameter exploration i.e. grid search: model architecture, regularization (I added dropout, but could further experiment).
1. Feature engineering - Engineering robust features, principal component analysis, representation learning etc. 
1. Type of approach: One could use other types of approaches such as LSTM for classification with SMOTE upsampling.
1. CONSULT THE LITERATURE!!! There is most likely tons of useful academic literature in this space.

