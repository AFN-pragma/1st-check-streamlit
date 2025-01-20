import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import requests
import io
from io import StringIO

#https://drive.google.com/file/d/1ioUQlMK5i53ugHxlEe7gqUD4k4_gaAkm/view?usp=drive_link

DATA = pd.read_csv('https://drive.usercontent.google.com/download?id=1ioUQlMK5i53ugHxlEe7gqUD4k4_gaAkm&export=download&authuser=0&confirm=t'.format(url.split('/')[-2]))


#url='https://drive.google.com/file/d/0B6GhBwm5vaB2ekdlZW5WZnppb28/view?usp=sharing'

#DATA = pd.read_csv("Expresso.csv")
#DATA 
#file_url = "https://drive.google.com/uc?id=1ioUQlMK5i53ugHxlEe7gqUD4k4_gaAkm"
#response = requests.get(file_url)
#dataset = io.StringIO(response.text)

# Lire le CSV dans un DataFrame
#DATA = pd.read_csv(dataset, on_bad_lines='warn')

#DATA = pd.read_csv('Expresso.csv', nrows=900000) #usecols=['x1', 'x2', 'x3']
st.dataframe(df)
st.write("Informations sur les données :")
buffer = io.StringIO()
DATA.info(buf=buffer)
st.text(buffer.getvalue())


st.write(DATA.info())
st.write(DATA.shape)
st.write(DATA.describe())
st.write(DATA.isnull().sum())

duplicate_mask = DATA.duplicated()
duplicates = DATA[duplicate_mask]
print(duplicates)

data = DATA.drop_duplicates()
data = DATA.fillna(0)
data
from sklearn.linear_model import LogisticRegression

x = data.drop(["user_id", "REGION", "TENURE", "MRG", "TOP_PACK"], axis = 1)
y = data["CHURN"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
logreg = LogisticRegression(max_iter= 1000)
logreg.fit(x_train, y_train)
y_pred  = logreg.predict(x_test)
le_score_est_de=accuracy_score(y_test, y_pred)
le_score_est_de 
st.title("Expresso Churn Prediction")
st.write(f"nous avons une précision de: {le_score_est_de}")
