# Diabetes-Prediction

Problem Statement
The aim of this guide is to build a classification model to detect diabetes. We will be using the diabetes dataset which contains 768 observations and 9 variables, as described below:
⦁	pregnancies - Number of times pregnant.
⦁	glucose - Plasma glucose concentration.
⦁	diastolic - Diastolic blood pressure (mm Hg).
⦁	triceps - Skinfold thickness (mm).
⦁	insulin - Hour serum insulin (mu U/ml).
⦁	bmi – Basal metabolic rate (weight in kg/height in m).
⦁	dpf - Diabetes pedigree function.
⦁	age - Age in years.
⦁	outcome - “1” represents the presence of diabetes while “0” represents the absence of it. This is the target variable.
Evaluation Metric
We will evaluate the performance of the model using accuracy, which represents the percentage of cases correctly classified.
Mathematically, for a binary classifier, it's represented as accuracy = (TP+TN)/(TP+TN+FP+FN), where:
⦁	True Positive, or TP, are cases with positive labels which have been correctly classified as positive.
⦁	True Negative, or TN, are cases with negative labels which have been correctly classified as negative.
⦁	False Positive, or FP, are cases with negative labels which have been incorrectly classified as positive.
⦁	False Negative, or FN, are cases with positive labels which have been incorrectly classified as negative.
Steps
In this guide, we will follow the following steps:
Step 1 - Loading the required libraries and modules.
Step 2 - Reading the data and performing basic data checks.
Step 3 - Creating arrays for the features and the response variable.
Step 4 - Creating the training and test datasets.
Step 5 - Building , predicting, and evaluating the neural network model.
The following sections will cover these steps.

Step 1 - Loading the Required Libraries and Modules
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
Step 2 - Reading the Data and Performing Basic Data Checks
The first line of code reads in the data as pandas dataframe, while the second line prints the shape - 768 observations of 9 variables. The third line gives the transposed summary statistics of the variables.
Looking at the summary for the 'outcome' variable, we observe that the mean value is 0.35, which means that around 35 percent of the observations in the dataset have diabetes. Therefore, the baseline accuracy is 65 percent and our neural network model should definitely beat this baseline benchmark.
df = pd.read_csv('diabetes.csv') 
print(df.shape)
df.describe().transpose()
Step 3 - Creating Arrays for the Features and the Response Variable
The first line of code creates an object of the target variable called 'target_column'. The second line gives us the list of all the features, excluding the target variable 'outcome, while the third line normalizes the predictors.
The fourth line displays the summary of the normalized data. We can see that all the independent variables have now been scaled between 0 and 1. The target variable remains unchanged.
target_column = ['Outcome'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()
Step 4 - Creating the Training and Test Datasets
The first couple of lines of code below create arrays of the independent (X) and dependent (y) variables, respectively. The third line splits the data into training and test dataset, and the fourth line prints the shape of the training and the test data.
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); 
print(X_test.shape)
Step 5 - Building, Predicting, and Evaluating the Neural Network Model
In this step, we will build the neural network model using the scikit-learn library's estimator object, 'Multi-Layer Perceptron Classifier'. The first line of code (shown below) imports 'MLPClassifier'.
The second line instantiates the model with the 'hidden_layer_sizes' argument set to three layers, which has the same number of neurons as the count of features in the dataset. We will also select 'relu' as the activation function and 'adam' as the solver for weight optimization. To learn more about 'relu' and 'adam', please refer to the Deep Learning with Keras guides.
The third line of code fits the model to the training data, while the fourth and fifth lines use the trained model to generate predictions on the training and test dataset, respectively.
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
Once the predictions are generated, we can evaluate the performance of the model. Being a classification algorithm, we will first import the required modules, which is done in the first line of code below. The second and third lines of code print the confusion matrix and the confusion report results on the training data.
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
The above output shows the performance of the model on training data. The accuracy and the F1 score is around 0.78 and 0.77, respectively. Ideally, the perfect model will have the value of 1 for both these metrics, but that is next to impossible in real-world scenarios.
The next step is to evaluate the performance of the model on the test data that is done with the lines of code below.
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
The above output shows the performance of the model on test data. The accuracy and F1 scores both around 0.75.
Conclusion
In this example, you have learned about building a neural network model using scikit-learn. The guide used the diabetes dataset and built a classifier algorithm to predict the detection of diabetes.
Our model is achieving a decent accuracy of 78 percent and 75 percent on training and test data, respectively. We observe that the model accuracy is higher than the baseline accuracy of 66 percent. The model can be further improved by doing cross-validation, feature engineering, or changing the arguments in the neural network estimator.
Note that we have built a classification model in this guide. However, building the regression model also follows the same structure, with a couple of adjustments. The first being that instead of the estimator 'MLPClassifier', we will instantiate the estimator 'MLPRegressor'. The second adjustment is that, instead of using accuracy as the evaluation metric, we will use RMSE or R-squared value for model evaluation.

