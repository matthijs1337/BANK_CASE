import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("bank.csv", sep=';')
df = bank.drop_duplicates
df = bank.rename(columns={'y', 'target'})
label_encoder = LabelEncoder()
df['job_encoded'] = label_encoder.fit_transform(df['job'])


# Train een logistisch regressiemodel met de dataset
X = df[['campaign', 'pdays']]
y = df['target']
model = LogisticRegression()
model.fit(X, y)

# Streamlit-app
st.title('Ja/Nee Voorspellingsdashboard')

# Dropdown-menu's voor variabelen
campaign = st.selectbox('Selecteer campaign:', df['campaign'].unique())
pdays = st.selectbox('Selecteer pdays:', df['pdays'].unique())

# Maak een voorspelling met het model op basis van de geselecteerde waarden
prediction = model.predict([[job, loan]])

# Toon het voorspelde resultaat
st.write(f"Voorspelling: {prediction[0]}")
