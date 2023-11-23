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

#duration
df.loc[df['duration'] < 249, 'DurationGroup'] = '0-249'
df.loc[(df['duration'] >= 250) & (df['duration'] < 499), 'DurationGroup'] = '250-499'
df.loc[(df['duration'] >= 500) & (df['duration'] < 749), 'DurationGroup'] = '500-749'
df.loc[(df['duration'] >= 750) & (df['duration'] < 999), 'DurationGroup'] = '750-999'
df.loc[df['duration'] >= 1000, 'DurationGroup'] = '1000+'

# Labelcodering toepassen
label_encoder = LabelEncoder()
df['age_encoded'] = label_encoder.fit_transform(df['AgeGroup'])
df['job_encoded'] = label_encoder.fit_transform(df['job'])
df['marital_encoded'] = label_encoder.fit_transform(df['marital'])
df['education_encoded'] = label_encoder.fit_transform(df['education'])
df['contact_encoded'] = label_encoder.fit_transform(df['contact'])
df['month_encoded'] = label_encoder.fit_transform(df['month'])
df['duration_encoded'] = label_encoder.fit_transform(df['DurationGroup'])



# Train een logistisch regressiemodel met de dataset
X = df[['age_encoded', 'job_encoded',
        'marital_encoded', 'contact_encoded'
        'education_encoded', 'contact_encoded'
        'month_encoded', 'duration_encoded'
       ]]
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
st.write(f"Voorspelling: {prediction[0]}")
