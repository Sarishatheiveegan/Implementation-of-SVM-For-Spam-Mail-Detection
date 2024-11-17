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
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

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

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Encoding:
![328234656-ed87456c-9dd8-418d-a960-1abad11477f2](https://github.com/user-attachments/assets/26f3b824-137a-4804-8660-5c21ddc4b5e8)
### Head():
![328234925-8e2c3fec-2fe3-40c3-923a-1a1c3719e734](https://github.com/user-attachments/assets/620c8737-1e1b-487b-a5ed-5fa009d81811)
### Info():
![328235099-b48518c5-c983-44d3-9cc2-14924033aa91](https://github.com/user-attachments/assets/7ce2aa23-9c90-4547-8df7-7c454b826c74)
### isnull().sum():
![328235367-50754f89-e886-48c3-a285-44b76317b605](https://github.com/user-attachments/assets/f5f51e0f-dee0-45b4-8d77-ae9a1ea14264)
### Prediction of y:
![328235504-8f3a2d63-9aa6-4da2-95c4-d53b87fde998](https://github.com/user-attachments/assets/44c56623-f6ca-406d-a830-f8073a86fa76)
### Accuracy:
![328235573-d1dcce16-dc32-4ec2-a042-ce25bee461da](https://github.com/user-attachments/assets/5aa69037-c14d-4262-a10f-bde6fc5afa50)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
