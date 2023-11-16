import streamlit as st
import pandas as pd
pip install -U scikit-learn
import sklearn


data = {'Feature1': [1, 2, 3, 4, 5],'Feature2': [2, 3, 4, 5, 6],'Target': ['Nee', 'Nee', 'Ja', 'Nee', 'Ja']}
df = pd.DataFrame(data)

# Train een logistisch regressiemodel met de dataset
X = df[['Feature1', 'Feature2']]
y = df['Target']
model = LogisticRegression()
model.fit(X, y)

# Streamlit-app
st.title('Ja/Nee Voorspellingsdashboard')

# Dropdown-menu's voor variabelen
feature1 = st.selectbox('Selecteer Feature1:', df['Feature1'].unique())
feature2 = st.selectbox('Selecteer Feature2:', df['Feature2'].unique())

# Maak een voorspelling met het model op basis van de geselecteerde waarden
prediction = model.predict([[feature1, feature2]])

# Toon het voorspelde resultaat
st.write(f"Voorspelling: {prediction[0]}")
