# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MARINO SARISHA T
RegisterNumber:  212223240084
*/

import chardet
with open('spam.csv','rb') as file:
    result = chardet.detect(file.read(10000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()
data.info()
data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc

```

## Output:
![Screenshot 2024-12-14 003451](https://github.com/user-attachments/assets/c37558da-3b85-49e1-bc65-1181baae0fa7)
![Screenshot 2024-12-14 003507](https://github.com/user-attachments/assets/355c5cf7-500c-4071-9ba1-b923db596838)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
