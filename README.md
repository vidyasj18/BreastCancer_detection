# BreastCancer_detection
## Project Overview
This project focusses on classifying malignant and benign tumor from the dataset UC Irvine Machine Learning Repository website, using a classification machine learning algorithm called "Logistic Regression". The goal is to accurately classify the tomors from dataset by training the model.

## Dataset
https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
This dataset contains 684 samples each has measurements/values for the 9 cell properties defined.
### Cell Properties
- Clump thickness
- Uniformity of cell size
- Uniformity of cell shape
- Marginal adhesion
- Single epithelial cell size
- Bare nuclei
- Bland chromatin
- Normal nucleoli
- Mitoses
### Class Labels
2 - benign
4 - malignant

## Libraries used
NumPy and Pandas : For numerical operations and data manipulation                                                                                         
Matplotlib : For data visualization and plotting metrics                        
scikit-learn : For importing algorithms and tools

## Data preprocessing
### Dataset splitting
The dataset is split into training (80%) and validation (20%) sets:                                          
