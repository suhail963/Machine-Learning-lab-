import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

data = pd.read_csv("../datasets/headbrain.csv")

X = data["Head Size(cm^3)"]

y = data["Brain Weight(grams)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 

random_state=42)

model = LinearRegression()

model.fit(np.expand_dims(X_train,axis=1),np.expand_dims(y_train,axis=1))

predictions = model.predict(np.expand_dims(X_test,axis=1))

plt.figure(figsize=(8,5))

plt.scatter(X_test,y_test,color="r",label='Scatter Plot')

plt.plot(X_test,predictions,color="b",label='Regression Line')

plt.xlabel('Head Size in cm3')

plt.ylabel('Brain Weight in grams')

plt.legend()


r2_score = 

model.score(np.expand_dims(X_test,axis=1),np.expand_dims(y_test,axis=1))

mse = mean_squared_error(y_test,predictions)

print(f"RMSE VALUE : {np.sqrt(mse)}\nR2 SCORE : {r2_score}")

output:

RMSE VALUE : 68.91317515113433

R2 SCORE : 0.6692497355337241
#logistic
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

titanic_data = pd.read_csv("../datasets/titanic.csv")
#plot1
sns.heatmap(titanic_data.isnull(),cbar=False)
#plot2
sns.countplot(x='Survived', hue='Sex', data=titanic_data)
#plot3
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
#another tab
def get_age(row):

 if pd.isnull(row["Age"]):

 same_class = titanic_data[titanic_data["Pclass"] == row["Pclass"]]

 return same_class["Age"].mean()

 else:

 return row["Age"]

titanic_data["Age"] = titanic_data[["Age","Pclass"]].apply(get_age ,axis=1)

titanic_data.drop("Cabin",axis=1,inplace=True)

titanic_data.dropna(inplace=True)

sns.heatmap(titanic_data.isnull(),cbar=False)

sex_data = pd.get_dummies(titanic_data["Sex"]).astype(int)

embarked_data = pd.get_dummies(titanic_data['Embarked']).astype(int)

titanic_data = pd.concat([titanic_data, sex_data, embarked_data], axis = 1)

titanic_data.head(5)
titanic_data.drop(['Name', 'PassengerId', 'Ticket', 'Sex', 'Embarked'], axis = 1, inplace =True)
X = titanic_data.drop("Survived",axis = 1)

y = titanic_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

model = LogisticRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("************Classification Report************\n")

print(classification_report(y_test, predictions))

print("***********Confusion Matrix***************")

print(confusion_matrix(y_test, predictions))


