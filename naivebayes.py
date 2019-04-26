import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

species = pd.read_csv("iris.csv")
species.head()
print(species)

number = LabelEncoder()
species['sepal_length'] = number.fit_transform(species['sepal_length'])
species['sepal_width'] = number.fit_transform(species['sepal_width'])
species['petal_length'] = number.fit_transform(species['petal_length'])
species['petal_width'] = number.fit_transform(species['petal_width'])
species['species'] = number.fit_transform(species['species'])
print("Encoded values\n",species)

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = "species"

features_train, features_test, target_train, target_test = train_test_split(species[features],species[target],test_size = 0.33,random_state = 54)

model = GaussianNB()
model.fit(features_train, target_train)

pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print("Accuracy in % is:",accuracy*100)
ans=model.predict([[5,3,1,2]])
if(ans==0):
	print("Setosa")
elif(ans==1):
	print("versicolor")
else:
	print("virginica")
