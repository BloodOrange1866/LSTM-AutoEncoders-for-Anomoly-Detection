# LSTM-Autoencoder for Outlier Detection

## Background

In this project, an LSTM Autoencoder is trained and applied in the context of outlier detection. The dataset used for this task is the [Ozone Level Detection dataset](https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection). The task is to use weather and enviromental data to predict an extreme weather event.

## Why anomoly detection, specifically an LSTM Autoencoder?

Due to the limited number of extreme weather events, anomoly detection in this context serves as the apropriate tool to use. A model can be constructed which learns from the normal weather events, and applied with the aim of recognising extreme events as anomolies. 


The core idea behind an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) is to learn a representation can be used to reconstruct the original input sequence. In the context of our problem, detecting extreme weather events, we want to train a model to reconstruct normal weather events which ultimately leads to a learnt model able to detect ‘normality’ as this is all the model has seen. 

In the cases of extreme weather events, the model has not ‘seen’ these types of sequences before, and ultimately has trouble reconstructing the sequence; this leads to a high reconstruction error. The reconstruction error then acts as a metric whereby we can set some threshold, i.e. if reconstruction error is > some_threshold must be an extreme weather event.

There are ways to set the threshold algorithmically, for now, I have used manual inspection (see below for further details).

Why an [LSTM-autoencoder](https://blog.keras.io/building-autoencoders-in-keras.html)? Due to weather events exhibiting a temporal nature, an LSTM can make use of the long-range dependencies inherit in the dataset. This approach assumes there is a temporal nature / ramp-up before an extreme weather event.

## What about other approaches?

I did ponder other approaches, rather than take the non-parametric discriminative route, one could use a generative approach to model the underlying distribution such as a [variational autoencoder LSTM](https://towardsdatascience.com/time-series-generation-with-vae-lstm-5a6426365a1c) which would additionally provide certainty estimates. Besides this, a classical approach could also be to construct a distribution of normal weather events and use this as a reference when testing for extreme weather events. 

[LSTM-AE](figures/lstm-ae.png?raw=true "Optional Title")
<img src="/figures/lstm-ae.png"> 