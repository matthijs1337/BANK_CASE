#Packages
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

#Data inladen
df = pd.read_csv("bank.csv", sep=';')

#Data cleaning
df = df.drop_duplicates()
value_to_delete = 'unknown'
df = df[(df['age'] != value_to_delete) &
                  (df['job'] != value_to_delete) &
                  (df['marital'] != value_to_delete) &
                  (df['education'] != value_to_delete) &
                  (df['contact'] != value_to_delete) &
                  (df['month'] != value_to_delete) &
                  (df['duration'] != value_to_delete) &
                  (df['campaign'] != value_to_delete) &
                  (df['pdays'] != value_to_delete) &
                  (df['previous'] != value_to_delete)]

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
    st.title('Voorspellingsmodel succesvolle banklening')
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

def plot_bar_charts():
    #figuur 1 leeftijdsgroep
    fig1 = px.bar(df, x='y', color="AgeGroup", barmode="group")
    st.plotly_chart(fig1)
    
    #figuur 2 lengte van de call
    fig2 = px.bar(df, x='y', color="DurationGroup", barmode="group")
    st.plotly_chart(fig2)
    
    #figuur 3 opleidingsniveau
    fig3 = px.bar(df, x='y', color="education", barmode="group")
    st.plotly_chart(fig3)
  
    #figuur 4 werk
    fig4 = px.bar(df, x='y', color="job", barmode="group")
    st.plotly_chart(fig4)

    #figuur 5 huwelijk
    fig5 = px.bar(df, x='y', color="marital", barmode="group")
    st.plotly_chart(fig5)


def plot_bar_charts2():
    #figuur 6 contact
    fig6 = px.bar(df, x='y', color="contact", barmode="group")
    st.plotly_chart(fig6)

    #figuur 7 maand
    fig7 = px.bar(df, x='y', color="month", barmode="group")
    st.plotly_chart(fig7)

    #figuur 8 campaign
    fig8 = px.bar(df, x='y', color="campaign", barmode="group")
    st.plotly_chart(fig8)
    
    #figuur 9 pdays
    fig9 = px.bar(df, x='y', color="pdays", barmode="group")
    st.plotly_chart(fig9)

    #figuur 10 
    fig10 = px.bar(df, x='y', color="previous", barmode="group")
    st.plotly_chart(fig10)
    
def tab_two():
    st.title('Plots')
    plot_bar_charts()

def tab_three():
    st.title('Plots 2')
    plot_bar_charts2()

def main():

    tabs = ["Tab 1", "Tab 2", "Tab 3"]
    choice = st.sidebar.selectbox("Select Tab", tabs)

    if choice == "Tab 1":
        tab_one()
    elif choice == "Tab 2":
        tab_two()
    elif choice == "Tab 3":
        tab_three()

if __name__ == "__main__":
    main()
