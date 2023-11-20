import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("bank.csv", sep=';')
df = df.drop_duplicates()

#categoriseren van data
#leeftijd
df.loc[df['age'] < 30, 'AgeGroup'] = '17-29'
df.loc[(df['age'] >= 30) & (df['age'] < 50), 'AgeGroup'] = '30-49'
df.loc[(df['age'] >= 50) & (df['age'] < 65), 'AgeGroup'] = '50-64'
df.loc[df['age'] >= 65, 'AgeGroup'] = '65+'

#opleiding


# Labelcodering toepassen
label_encoder = LabelEncoder()
df['job_encoded'] = label_encoder.fit_transform(df['job'])
df['age_encoded'] = label_encoder.fit_transform(df['AgeGroup'])
df['education_encoded'] = label_encoder.fit_transform(df['education'])


# Train een logistisch regressiemodel met de dataset
X = df[['job_encoded', 'pdays', 'age_encoded', 'education_encoded']]
y = df['y']
model = LogisticRegression()
model.fit(X, y)
# Streamlit-app
st.title('Ja/Nee Voorspellingsdashboard')
# Dropdown-menu's voor variabelen
job_encoded = st.selectbox('Selecteer baan:', df['job_encoded'].unique())
age_encoded = st.selectbox('Selecteer leeftijdsgroep:', df['age_encoded'].unique())
education_encoded = st.selectbox('Selecteer opleidingsniveau:', df['education_encoded'].unique())
pdays = st.selectbox('Selecteer pdays:', df['pdays'].unique())
# Maak een voorspelling met het model op basis van de geselecteerde waarden
prediction = model.predict([[job_encoded, age_encoded, education_encoded, pdays]])
# Toon het voorspelde resultaat
st.write(f"Voorspelling: {prediction[1]}")
