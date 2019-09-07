# Signal detection project
Goal of the project : developping a system capable of classification, localization and detection of
radiocommunication signals on spectrograms
## Method
Overfeat <a href="https://arxiv.org/pdf/1312.6229.pdf">paper</a>

## Classification network
<p>
The classification network consists of two parts: a first called feature extractor and a second one that classifies.
 <ul><li>the feature extractor is composed of a series of 3 convolution layers (the first two are followed by a max pooling layer)
<li>the last part of the classifier consist of fully connected layers that generate a vector of 12 components (the number of classes).
</ul> 
</p>
<p>
The output of all layers(except the last one) is normalized using the batch normalization technique (in order to
increase the stability of the neural network during training)
</p>
<p>In addition, layers 4 and 5 (fully
connected) have a dropout of 40 percent. This eliminates weights
randomly during training, which reduces the number of parameters to
to train network and avoid the phenomenon of overfitting.
</p>
<p align="center"><img src="img/classifier.PNG" height=250 width=450></img></p>

## Regression network
<p align="center"><img src="img/regressor.PNG" height=200 width=400></img></p>

## HMI

<p>The human-machine interface allows to select a spectrogram in the file system and can display the detected and localized signals with
bounding boxes and indicating the class of the selected spectrogram</p>
<p align="center"><img src="img/hmi.PNG" height=300 width=400></img></p>
