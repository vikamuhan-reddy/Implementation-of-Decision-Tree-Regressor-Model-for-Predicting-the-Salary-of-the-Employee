# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import Libraries: Import necessary libraries for data manipulation and machine learning.
2.Read Data: Load the dataset from the CSV file.
3.Explore Data: Print out information about the dataset like its structure and whether there are any missing values.
4.Encode Categorical Data: Convert categorical data into numerical format for machine learning algorithms.
5.Split Data: Divide the dataset into features (inputs) and target variable (output), then split them into training and testing sets.
6.Model Training: Create a decision tree model and train it using the training data.
7.Make Predictions: Use the trained model to predict salaries based on position and level for the test data.
8.Evaluate Model: Assess the model's performance using metrics like Mean Squared Error (MSE) and R-squared score.
9.Make Predictions on New Data: Apply the trained model to predict salary for a new data point with position 5 and level 6.

## Program:
```py
'''Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vikamuhan Reddy.N
RegisterNumber: 212223240181
'''
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Salary.csv")
print(data)
print(data.info())
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
print(data.head())
x=data[['Position','Level']]
y=data['Salary']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_predict)
print(mse)
r2=metrics.r2_score(y_test,y_predict)
print(r2)
print(dt.predict([[5,6]]))
```

## Output:
![Screenshot 2024-04-06 114237](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144928933/515340a3-257c-4b60-bbef-c6fff920b19e)
![Screenshot 2024-04-06 114247](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144928933/aa997a38-1b53-488d-bbbd-d59a19ee0bcc)
![Screenshot 2024-04-06 114321](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144928933/22f22e34-ee78-4dae-b6ed-c19c9325fc88)
![Screenshot 2024-04-06 114328](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144928933/c24281d0-f96a-4a1b-81de-90f970e841e1)
![Screenshot 2024-04-06 114339](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144928933/121ca0d2-ab49-4a58-91d9-b4abd40b6768)
![Screenshot 2024-04-06 114353](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144928933/8ff168a4-628c-4e1f-beff-cbf21cb446f4)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
