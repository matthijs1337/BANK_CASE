import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

bank = pd.read_csv("bank.csv", sep=';')
bank = bank.rename(columns={'y', 'target'})

# Train een logistisch regressiemodel met de dataset
X = bank[['job', 'loan']]
y = bank['target']
model = LogisticRegression()
model.fit(X, y)

# Streamlit-app
st.title('Ja/Nee Voorspellingsdashboard')

# Dropdown-menu's voor variabelen
job = st.selectbox('Selecteer job:', bank['job'].unique())
loan = st.selectbox('Selecteer loan:', bank['loan'].unique())

# Maak een voorspelling met het model op basis van de geselecteerde waarden
prediction = model.predict([[job, loan]])

# Toon het voorspelde resultaat
st.write(f"Voorspelling: {prediction[0]}")
