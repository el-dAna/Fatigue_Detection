# Fatigue_Detection
This repository contains two classification models for physiological signal classification into 4 classes(Relax, PhysicalStress, CognitiveStress, EmotionalStress)

Before anything, remember to change working directory to Fatigue_Detection.

The dataset is contained in HealthySubjectsBiosignalsDataSet directory.
Data for the model is preprocessed using preprocess.py file. This file calls all functions from the preprocessingfucntions.py file
Run !python preprocess.py to preprocess the data and save the results in saved_vars.py to be accessed by the models.

The contents in saved_vars.py file in a cloned repository have been proprocessed already. You can run the line above to get a fresh copy anyway.

This repositiry has two models, RNN model and a CNN model(Transfer learning using InceptionV3 imagenet weights).
Run

!python RNN_train.py #to train the RNN model

or

!python WL_train.py # to train the WL(WaveLet Transform) model.

The architecture of each model located in RNN_model.py and WL_model.py respectively.
These models access common functions from the common_functions.py file.
The WL_fucntions contain functions specific to the Wavelet Transform CNN model.

The models are each not in a satisfactory state.
The RNN model has high accuracy but poor performance on the confusion matrix
The wL model has low accuracy.
