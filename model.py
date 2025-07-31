#importing the packages in the dataset
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import os

#Loading the dataset

data = pd.read_csv("Crop_recommendation.csv")

# data.head()

#Cleaning the dataset

# print(
#    "--------Shape of the dataset-------- \n",
#    data.shape ,"\n""\n"
# )

# print(
#    "--------Null values in dataset-------- \n",
#    data.isnull().sum() ,"\n""\n"
#    )

# print(
#    "--------Nan values in dataset-------- \n",
#    data.isna().sum() ,"\n""\n"
#    )
# print(
#    "--------Data types in dataset-------- \n",
#    data.dtypes ,"\n""\n"
#    )

x = data.iloc[:, :-1] #features
y = data.iloc[:,  -1] #labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# x_train.head()

# x_test.head()

# y_train.head()

# y_test.head()

model = RandomForestClassifier()

model.fit(x_train, y_train)

# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)

# print("Accuracy", accuracy)

# new_data = pd.DataFrame([{
#     'N': 36,
#     'P': 58,
#     'K': 25,
#     'temperature': 28.66024,
#     'humidity': 59.399136,
#     'ph': 6.5,
#    'rainfall': 100.0
# }])

# predicted_crop = model.predict(new_data)
# print("🌾 Predicted crop:", predicted_crop)

pickle.dump(model, open("model.pkl", "wb"))