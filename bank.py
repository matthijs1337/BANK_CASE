import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("bank.csv", sep=';')
df = df.drop_duplicates()

# Label encoding
label_encoder = LabelEncoder()
df['job_encoded'] = label_encoder.fit_transform(df['job'])
df['pdays_encoded'] = label_encoder.fit_transform(df['pdays'])  # Encode pdays

# Train a logistic regression model with the dataset
X = df[['job_encoded', 'pdays_encoded']]  # Use the encoded pdays
y = df['y']
model = LogisticRegression()
model.fit(X, y)

# Streamlit app
st.title('Yes/No Prediction Dashboard')

# Dropdown menus for variables
job_options = df['job'].unique()
job_encoded = st.selectbox('Select job:', job_options)

pdays_options = df['pdays'].unique()
pdays_encoded = st.selectbox('Select pdays:', pdays_options)

# Encode selected pdays value
pdays_encoded_selected = label_encoder.transform([pdays_encoded])[0]

# Make a prediction with the model based on the selected values
prediction = model.predict([[job_encoded, pdays_encoded_selected]])

# Show the predicted result
st.write(f"Prediction: {prediction[0]}")
