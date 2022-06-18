# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
Original data(first five columns):

![image](https://user-images.githubusercontent.com/67967960/174423116-7ecd2633-d1a6-4490-ab7c-4453f81d20e5.png)

Data after dropping unwanted columns(first five):

![image](https://user-images.githubusercontent.com/67967960/174423133-74b1936e-12f7-4217-a771-22de518a6587.png)

Checking the presence of null values:

![image](https://user-images.githubusercontent.com/67967960/174423144-c49dd880-f853-477f-afc7-526c5ff375e5.png)

Checking the presence of duplicated values:

![image](https://user-images.githubusercontent.com/67967960/174423157-c3e9bcb9-fa39-4a8c-ac89-732a81c916d3.png)

Data after Encoding:

![image](https://user-images.githubusercontent.com/67967960/174423176-d60b83f7-55ea-4634-8cdf-41b917a9748a.png)

X Data:

![image](https://user-images.githubusercontent.com/67967960/174423190-32b7d11f-ae71-48fd-9043-cc25a486b8d1.png)

Y Data:

![image](https://user-images.githubusercontent.com/67967960/174423203-740c7cff-b43b-4c35-8294-a2b28ea77f03.png)

Predicted Values:

![image](https://user-images.githubusercontent.com/67967960/174423220-58a4de8b-0dd2-4558-976c-828c844033d6.png)

Accuracy Score:

![image](https://user-images.githubusercontent.com/67967960/174423240-60b822b9-5b6c-43ce-ba10-da784dd48f2b.png)

Confusion Matrix:

![image](https://user-images.githubusercontent.com/67967960/174423265-f98345d0-9bd4-487a-bf08-b6440ff0d6cc.png)

Classification Report:

![image](https://user-images.githubusercontent.com/67967960/174423278-55ee9fea-3d76-477d-a9f7-d8c0d5601845.png)

Predicting output from Regression Model:

![image](https://user-images.githubusercontent.com/67967960/174423291-7134e919-644c-4ca6-8ca2-13ef37b05eca.png)













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
