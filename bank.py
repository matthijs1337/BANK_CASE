#Packages
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
#Data inladen
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

def tab_one():
    
            # Train een logistisch regressiemodel met de dataset
            X = df[['age_encoded', 'job_encoded','marital_encoded', 'education_encoded', 'contact_encoded', 'month_encoded', 'duration_encoded', 'campaign', 'pdays','previous']]
            y = df['y']
            model = LogisticRegression()
            model.fit(X, y)
            # Streamlit-app
            st.title('Ja/Nee Voorspellingsdashboard')
            # Dropdown-menu's voor variabelen
            age_encoded = st.selectbox('Selecteer leeftijdsgroep:', df['age_encoded'].unique())
            job_encoded = st.selectbox('Selecteer baan:', df['job_encoded'].unique())
            marital_encoded = st.selectbox('Selecteer relatiestatus:', df['marital_encoded'].unique())
            education_encoded = st.selectbox('Selecteer opleidingsniveau:', df['education_encoded'].unique())
            contact_encoded = st.selectbox('Selecteer contact:', df['contact_encoded'].unique())
            month_encoded = st.selectbox('Selecteer maand:', df['month_encoded'].unique())
            duration_encoded = st.selectbox('Selecteer duration:', df['duration_encoded'].unique())
            campaign = st.selectbox('Selecteer campaign:', df['campaign'].unique())
            pdays = st.selectbox('Selecteer pdays:', df['pdays'].unique())
            previous = st.selectbox('Selecteer previous:', df['previous'].unique())
            # Maak een voorspelling met het model op basis van de geselecteerde waarden
            prediction = model.predict([[age_encoded,job_encoded,marital_encoded,education_encoded,contact_encoded,month_encoded,duration_encoded,campaign,pdays,previous]])
            # Toon het voorspelde resultaat
            st.write(f"Voorspelling: {prediction[0]}")

def tab_two():

            #Maken van Plots
def plot_bar_chart():
fig = px.bar(df, x='y', color="AgeGroup", barmode="group"), title='Bar Chart')
st.plotly_chart(fig)
st.write("Content of Tab 2")
    

def main():

    tabs = ["Tab 1", "Tab 2"]
    choice = st.sidebar.selectbox("Select Tab", tabs)

    if choice == "Tab 1":
        tab_one()
    elif choice == "Tab 2":
        tab_two()

if __name__ == "__main__":
    main()

