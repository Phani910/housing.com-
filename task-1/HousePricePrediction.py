import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # oye thingari bochu indaka try ches copy chesa diff ga vachindi okasari folder open chey new di chesta chudu ha sare chudu
import seaborn as sns
# hey mental code file eh csv kaadhu ha chudu em chudali

df = pd.read_csv(r"C:\Users\PHANISREE SAI\Documents\Interns\Housing.csv")
# FIRST 10 ROWS OF THE DATASET   
df.head(10)

# COLUMNS :
df.columns

# SIZE OF THE DATASET 
df.shape

# DATA TYPES OF THE COLUMNS OF THE DATASET 
df.info()

# CHECKING FOR NULL VALUES
df.isnull().sum()

# REMOVAL OF DUPLICATE VALUE 
counter = 0
rs,cs = df.shape
df.drop_duplicates(inplace=True)

if df.shape==(rs,cs):
    print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
else:
    print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {rs-df.shape[0]}')

# CONVERTING ALL OUR CATEGORICAL DATA COLUMNS TO NUMERIC FORM
from sklearn.preprocessing import LabelEncoder 
categ = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]
# Encode Categorical Columns
le = LabelEncoder()
df[categ] = df[categ].apply(le.fit_transform)

# AFTER CONVERTING OUR DATASET LOOKS LIKE THIS ...
df.head()

# CORRELATION BETWEEN THE COLUMNS
corr = df.corr()
plt.figure(figsize=(12,7))
sns.heatmap(corr,cmap='coolwarm',annot=True)

X = df.drop(['price'],axis=1)
y = df['price']
X

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# LENGTH OF X_train AND X_test
len(X_train),len(X_test)

# IMPORTING THE MODULE
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# FITTING THE DATA INTO THE MODEL
model.fit(X_train,y_train)

# PREDICTING THE OUTCOMES
y_predict = model.predict(X_test)
y_predict

from sklearn.metrics import r2_score,mean_absolute_error
score = r2_score(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)

score

mae

