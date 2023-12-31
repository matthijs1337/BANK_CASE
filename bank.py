#Packages
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
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

    #Converting dependent variable categorical to dummy
    bank = df.copy()
    y = pd.get_dummies(bank['y'], columns = ['y'], prefix = ['y'], drop_first = True)
      
    bank_client = bank.iloc[: , 0:7]
    # Label encoder order is alphabetical
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    bank_client['job']      = labelencoder_X.fit_transform(bank_client['job']) 
    bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital']) 
    bank_client['education']= labelencoder_X.fit_transform(bank_client['education']) 
    bank_client['default']  = labelencoder_X.fit_transform(bank_client['default']) 
    bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing']) 
    bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan']) 
    bank_related = bank.iloc[: , 7:11]
    # Label encoder order is alphabetical
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact']) 
    bank_related['month']       = labelencoder_X.fit_transform(bank_related['month']) 
    bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week']) 
    def duration(data):
    
        data.loc[data['duration'] <= 102, 'duration'] = 1
        data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
        data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
        data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
        data.loc[data['duration']  > 644.5, 'duration'] = 5
    
        return data
    duration(bank_related);
    bank_se = bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
    bank_o = bank.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
    bank_o['poutcome'].unique()
    bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
    bank_final= pd.concat([bank_client, bank_related, bank_se, bank_o], axis = 1)
    bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                         'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                         'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]
    X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.1942313295, random_state = 101)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    logmodel = LogisticRegression() 
    logmodel.fit(X_train,y_train)
    logpred = logmodel.predict(X_test)
    
    accuracy = round(accuracy_score(y_test, logpred), 2) * 100
    st.write("Nauwkeurigheid van de voorspelling:", accuracy, "%")
#Defineren van plots

def plot_bar_charts():
    #figuur 1 leeftijdsgroep
    #fig1 = px.bar(df, x='y', color="AgeGroup", barmode="group")
    #fig1 = px.bar(df, x='y', color="AgeGroup", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'AgeGroup' : 'Leeftijdscategorie', "count" : 'Aantal personen'})
    #st.plotly_chart(fig1)
    fig20 = px.pie(df, names='y', title="Percentage aan personen die wel of geen termijndeposito heeft afgesloten")
    fig20.update_layout(
    xaxis_title="Categorie",
    yaxis_title="Percentage",
    )
    st.plotly_chart(fig20)
  
    # Filter de dataset voor "yes" en "no"
    df_yes = df[df['y'] == 'yes']
    df_no = df[df['y'] == 'no']
  
    # Maak een subplots figuur met twee pie charts
    fig11 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig11.add_trace(go.Pie(labels=df_yes['AgeGroup'], hole=0.6), 1, 1)
    fig11.add_trace(go.Pie(labels=df_no['AgeGroup'], hole=0.6), 1, 2)
    fig11.update_layout(title_text="Leeftijdscategorie, Ja vs Nee")
    st.plotly_chart(fig11)
  
    #figuur 2 lengte van de call
    #fig2 = px.bar(df, x='y', color="DurationGroup", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'DurationGroup' : 'Duur van contact', "count" : 'Aantal personen'})
    #st.plotly_chart(fig2)
    fig12 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig12.add_trace(go.Pie(labels=df_yes['DurationGroup'], hole=0.6), 1, 1)
    fig12.add_trace(go.Pie(labels=df_no['DurationGroup'], hole=0.6), 1, 2)
    fig12.update_layout(title_text="Lengte van de call, Ja vs Nee")
    st.plotly_chart(fig12)
    
    #figuur 3 opleidingsniveau
    #fig3 = px.bar(df, x='y', color="education", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'education' : 'Hoogst afgeronde opleiding', "count" : 'Aantal personen'})
    #st.plotly_chart(fig3)
    fig13 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig13.add_trace(go.Pie(labels=df_yes['education'], hole=0.6), 1, 1)
    fig13.add_trace(go.Pie(labels=df_no['education'], hole=0.6), 1, 2)
    fig13.update_layout(title_text="Hoogst afgeronde opleiding, Ja vs Nee")
    st.plotly_chart(fig13)
  
    #figuur 4 werk
    #fig4 = px.bar(df, x='y', color="job", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'job' : 'Functietitel', "count" : 'Aantal personen'})
    #st.plotly_chart(fig4)
    fig14 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig14.add_trace(go.Pie(labels=df_yes['job'], hole=0.6), 1, 1)
    fig14.add_trace(go.Pie(labels=df_no['job'], hole=0.6), 1, 2)
    fig14.update_layout(title_text="Functietitel, Ja vs Nee")
    st.plotly_chart(fig14)
  
    #figuur 5 huwelijk
    #fig5 = px.bar(df, x='y', color="marital", barmode="group", labels ={"y": "Heeft de klant een termijndeposito afgesloten?", 'marital' : 'Huwelijksstatus', "count" : 'Aantal personen'})
    #st.plotly_chart(fig5)
    fig15 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig15.add_trace(go.Pie(labels=df_yes['marital'], hole=0.6), 1, 1)
    fig15.add_trace(go.Pie(labels=df_no['marital'], hole=0.6), 1, 2)
    fig15.update_layout(title_text="Huwelijksstatus, Ja vs Nee")
    st.plotly_chart(fig15)

def plot_bar_charts2():
    #figuur 6 contact
    fig6 = px.bar(df, x='y', color="contact", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'contact' : 'Communicatietype', "count" : 'Aantal personen'})
    st.plotly_chart(fig6)

    #figuur 7 maand
    fig7 = px.bar(df, x='y', color="month", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'month' : 'Maand', "count" : 'Aantal personen'} )
    st.plotly_chart(fig7)

    #figuur 8 campaign
    fig8 = px.bar(df, x='y', color="campaign", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'campaign' : 'Aantal uitgevoerde contacten vóór deze campagne en voor deze klant aantal uitgevoerde contacten tijdens deze campagne en voor deze klant (numeriek, inclusief laatste contact)', "count" : 'Aantal personen'})
    st.plotly_chart(fig8)
    
    #figuur 9 pdays
    fig9 = px.bar(df, x='y', color="pdays", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'pdays' : 'Aantal dagen dat is verstreken nadat de klant voor het laatst werd gecontacteerd vanuit een eerdere campagne', "count" : 'Aantal personen'})
    st.plotly_chart(fig9)

    #figuur 10 
    fig10 = px.bar(df, x='y', color="previous", barmode="group", labels ={"y" : "Heeft de klant een termijndeposito afgesloten?", 'previous' : 'Aantal uitgevoerde contacten vóór deze campagne en voor deze klant', "count" : 'Aantal personen'})
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
