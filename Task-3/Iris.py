import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = pd.read_csv(r"C:\Users\PHANISREE SAI\Desktop\BharatInterns\Ml\Task-3\iris.csv")
iris.head(10)
iris.shape
iris.columns

# CHECKING FOR NULL VALUES 
iris.isnull().sum()

# DROPPING THE UNNECESSARY
iris = iris.drop('Unnamed: 0',axis=1)
iris.head(10)
iris['Species'].value_counts()
n = len(iris[iris['Species'] == 'versicolor'])
print("No of Versicolor in Dataset:",n)
n1 = len(iris[iris['Species']=='setosa'])
print("No of setosa in dataset",n)
n2 = len(iris[iris['Species']=='virginica'])
print("No of virginica in dataset",n)
#Checking for outliars
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([iris['Sepal.Length']])
plt.figure(2)
plt.boxplot([iris['Sepal.Width']])
plt.show()
iris.hist()
plt.figure(figsize=(10,7))
plt.show()
sns.pairplot(iris,hue='Species')
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
train, test = train_test_split(iris, test_size = 0.3)
print(train.shape)
print(test.shape)
train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
train_y = train.Species

test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
test_y = test.Species
train_X
train_y
model1 = LogisticRegression()
model1.fit(train_X, train_y)
prediction = model1.predict(test_X)
prediction
print('Accuracy:',metrics.accuracy_score(prediction,test_y))
#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))
results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'KNN','Naive Bayes'],
    'Score': [0.9777,0.9777,0.9555,0.9555]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)