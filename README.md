# BreastCancer-Detection


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

simple table used to measure how well a classification model is performing. It compares the predictions made by the model with the actual results and shows where the model was right or wrong. It breaks down the predictions into four categories:

- True Positive (TP): The model correctly predicted a positive outcome i.e the actual outcome was positive.
  
- True Negative (TN): The model correctly predicted a negative outcome i.e the actual outcome was negative.
  
- False Positive (FP): The model incorrectly predicted a positive outcome i.e the actual outcome was negative. It is also known as a Type I error.
  
- False Negative (FN): The model incorrectly predicted a negative outcome i.e the actual outcome was positive. It is also known as a Type II error.

#### Confusion Matrix is found using the code

```

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

```

#### Output of confusion matrix

```

[[84  3]
 [ 3 47]]

```

#### Accuracy of the model using confusion matrix

It is calculated by the formula (TP + FP)/(Total number of y_test data).

```

accuracy = (84+47)/(84+47+3+3)

```

Total accuracy in this case = 0.9562043795620438.
(equivalent to 95.62%)


## K-Fold Cross Validation

#### Accuracy of model using K-Fold cross validation

```

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))

```

Accuracy : 96.70%                                                                                                              

Standard Deviation: 1.97 %

## Predict a new result

Code used to predict benign or malignant based on the input features.

#### Input Data:

```

input_data = {
    'mean radius': 15.0,
    'mean texture': 20.0,
    'mean perimeter': 90.0,
    'mean area': 600.0,
    'mean smoothness': 0.1,
    # Add other feature values here
}

```


#### Function used to Predict

```

def predict_breast_cancer(model, input_data, feature_names):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(input_df)
    if prediction[0] == 0:
        return "benign (not having breast cancer)."
    else:
        return "malignant (having breast cancer)."

```

