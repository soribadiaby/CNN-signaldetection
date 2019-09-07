# Signal detection project
Goal of the project : developping a system capable of classification, localization and detection of
radiocommunication signals on spectrograms
## Method
Overfeat <a href="https://arxiv.org/pdf/1312.6229.pdf">paper</a>
## Classification network
<p align="center"><img src="img/classifier.PNG" height=400 width=800></img></p>
## Regression network
<p align="center"><img src="img/regressor.PNG" height=400 width=800></img></p>
## HMI

The human-machine interface allows to select a spectrogram in the file system and can display the detected and localized signals with
bounding boxes and indicating the class of the selected spectrogram
<p align="center"><img src="img/hmi.PNG" height=400 width=400></img></p>
