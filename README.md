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

```
from sklearn.model_selection import train_test_split                                                                              
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

```
## Model Architechture
### Base Model
This project uses a Logistic Regression model.This is trained on the given dataset.

```

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

```


### Training of model
1. Trained using Logistic Regression imported from scikit-learn library.

2. Arguement inside the LogisticRegressor used is random_state = 0. random_state arguement is used to shuffle the data values inside the dataset.
Here, it is set to zero representing there is no shuffling in the data values as such.

3. The fit() method adjusts the model parameters based on the input data (X_train) and the target values (y_train).

## Prediction of Test set results

Model is used to predict the results of the test set.

```

y_pred = classifier.predict(X_test)

```

- This is done to understand how the model works. It is compare with y_test values later on to understand how accurate the model is and to know the differences between predicted and tested values.

## Confusion Matrix

simple table used to measure how well a classification model is performing. It compares the predictions made by the model with the actual results and shows where the model was right or wrong. This helps in understanding where the model is making mistakes so to improve it. It breaks down the predictions into four categories:

- True Positive (TP): The model correctly predicted a positive outcome i.e the actual outcome was positive.
  
- True Negative (TN): The model correctly predicted a negative outcome i.e the actual outcome was negative.
  
- False Positive (FP): The model incorrectly predicted a positive outcome i.e the actual outcome was negative. It is also known as a Type I error.
  
- False Negative (FN): The model incorrectly predicted a negative outcome i.e the actual outcome was positive. It is also known as a Type II error.
