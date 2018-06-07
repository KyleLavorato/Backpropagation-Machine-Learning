# Artificial Neural Network - Machine Learning

This repository houses two different Artificial Neural Network (ANN) experiments, where a complex Backpropagation network is used to classify entries in a data set, mased on a small training sample.

## Handwriting Analysis

This backpropagation network is hand coded in Python as a set of linked Backpropagation object nodes. The input dataset is the Optical Recognition of Handwritten Digits from the [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits). The data consists of many occurrences of handwritten digits from 0-9 to be classified by the ANN. The 32x32 digit grid is preprocessed into an 8x8 grid, where each 4x4 space in the original is compressed into the range 0-16, based on its activation. The preprocessed training data is then fed to the ANN, to greatly reduce processing time. Then once presented with testing data, the ANN can identify the numerical digit from the preprocessed handwritten data. The ANN source code is presented in `BackpropagationNetwork.py` and the results of the experiment are presented in `TechnicalReport.pdf`.

## Wine Quality Analysis

This backpropagation network takes use of the efficient built-in Backpropagation Nodes in the Matlab modeling software. The input data set is the Wine Quality Data from the [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). The data set consists of 4898 entries of red and white Vinho Verde wine samples, based on 12 attributes. Each sample also has a final quality score, that is dependent on those attributes. The goal of the experiment is to train the ANN to be able to correctly set a quality score for a wine sample, given the 12 attributes. To accomplish this, the training data was first preprocessed, to reduce training time and improve accuracy. The ANN was then tested on the remaining data to determine how often it was able to guess the wine quality correctly. For the full results of the experiment and applications of the research, see the technical report as `TechnicalReport.pdf`.

#### Citation

Thank you to the authors Cortez et al., 2009 for the Vino Verde data set and Alpaydin et al., 1998 for the Handwriting Data set. Their efforts in creating the data was invaluable in this research.