# ECG Classification App

This is a Streamlit app for classifying ECG data as either normal or abnormal (arrhythmia). 

## Overview

The ECG Classification App allows users to upload ECG data in CSV format and receive a classification prediction indicating whether the ECG is normal or abnormal. The app uses a pre-trained machine learning model to make predictions.

## Data Preprocessing

The model was trained on features extracted from regular ECG readings. These features include RR Intervals, Heartbeat Intervals features, Heart beats amplitude features and Morphology features. The model cannot be used to classify raw ECG data directly; it requires these preprocessed features.

## Acknowledgements

This project makes use of a dataset that was found on Kaggle. Please refer to the following paper for details of each feature and the methodology followed to generate these features from raw ECG data:
https://www.taylorfrancis.com/chapters/edit/10.1201/9781003028635-11/harnessing-artificial-intelligence-secure-ecg-analytics-edge-cardiac-arrhythmia-classification-sadman-sakib-mostafa-fouda-zubair-md-fadlullah
