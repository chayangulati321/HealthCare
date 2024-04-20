import os
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Care",
                   layout="wide",
                   page_icon="🏥")


# loading the saved models
data = pd.read_csv('hospital_visits_clean_data.csv')

data.drop(["Row_ID","Date of Admit","Date of Discharge","Doctor","Patient ID","Patient Name","Number of Patient Visits"], axis=1, inplace=True)
le_department_type = LabelEncoder()
data['Department Type'] = le_department_type.fit_transform(data['Department Type'])
le_patient_risk = LabelEncoder()
data['Patient Risk Profile'] = le_patient_risk.fit_transform(data['Patient Risk Profile'])
    
# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Hospital Branch', 'Department'])

X = data.drop('Revenue', axis=1)
y = data['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)


# model = pickle.load(open('Hospital_Revenue_Model.pkl', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Hospital Revenue',

                           ['Overview',
                            'Revenue Prediction'],
                           menu_icon='hospital-fill',
                           icons=['globe', 'currency-dollar'],
                           default_index=0)


# Overview Page
if selected == 'Overview':

    # page title
    st.title('Hospital Details')

    df = pd.read_csv('hospital_visits_clean_data.csv')
    st.text(" ")
    st.text("Row_ID: A unique identifier for each row in the dataset.")
    st.text("Date of Admit: The date when the patient was admitted to the hospital.")
    st.text("Date of Discharge: The date when the patient was discharged from the hospital.")
    st.text("Doctor: The name or identifier of the doctor responsible for the patient's care.")
    st.text("Hospital Branch: The specific branch or location of the hospital where the patient was admitted.")
    st.text("Department Type: The type or category of the department within the hospital.")
    st.text("Department: The name or identifier of the department within the hospital where the patient received treatment.")
    st.text("Patient ID: A unique identifier for each patient.")
    st.text("Patient Name: The name of the patient admitted to the hospital.")
    st.text("Patient Risk Profile: An assessment of the patient's risk level or medical condition.")
    st.text("Revenue: The amount of revenue generated from the patient's visit or treatment.")
    st.text("Minutes to Service: The duration in minutes from the time the patient arrived at the hospital to when they received service or treatment.")
    st.text("Number of Patient Visits: The total number of visits the patient made to the hospital.")
    st.text("Days in Hospital: The total number of days the patient spent in the hospital during their admission.")
    st.text(" ")
    preview = st.checkbox('Preview top 10 rows of the Data')

    if preview:
        st.write(df.head(10))

    st.text(" ")

    st.subheader('Description of the Hospital Dataset')
    st.write(df.describe())
    st.text(" ")
    #plt.figure(figsize=(8,3))
    st.subheader("Department Type wise Revenue")
    bar_chart1 = sns.barplot(x='Department Type', y='Revenue', data=df)
    st.pyplot(bar_chart1.get_figure())
    
    st.text(" ")
    st.text(" ")
    #plt.figure(figsize=(8,3))
    #plt.xticks(rotation=90)
    st.subheader("Department wise Days in Hospital")
    bar_chart2 = sns.barplot(x='Department', y='Days in Hospital', data=df)
    st.pyplot(bar_chart2.get_figure())
    
    
# Revenue Prediction Page
if selected == 'Revenue Prediction':

    # page title
    st.title('Hospital Revenue Prediction using ML')

    st.image('Hospital_Revenue_Image.png', caption='Hospital Revenue $')
    st.text(" ")

    risk_dict = {"Low":1,"High":0}
    feature_dict = {"General":0, "Intensive Care":1, "Labs":2, "Specialty":3}

    def get_value(val,my_dict):
	    for key,value in my_dict.items():
		    if val == key:
			    return value 

    def get_key(val,my_dict):
	    for key,value in my_dict.items():
		    if val == key:
			    return key

    def get_fvalue(val):
          feature_dict = {"General":0, "Intensive Care":1, "Labs":2, "Specialty":3}
          for key,value in feature_dict.items():
               if val == key:
                    return value

    col1, col2 = st.columns(2)

    with col1:
        department_type = st.selectbox("Department Type", tuple(feature_dict.keys()))

    with col2:
        patient_risk_profile = st.selectbox("Patient Risk Profile", tuple(risk_dict.keys()))

    with col1:
        minutes_to_service = st.number_input("Minutes to Service", min_value=0, max_value=500)

    with col2:
        days_in_hospital = st.number_input("Days in Hospital", min_value=0, max_value=30)

    with col1:
        hospital_branch = st.selectbox("Hospital Branch", data.columns[data.columns.str.startswith('Hospital Branch_')])

    with col2:
        department = st.selectbox("Department", data.columns[data.columns.str.startswith('Department_')])

    # creating a button for Prediction
    st.text(" ")
    if st.button('Predict'):

        # Make prediction
        input_data = pd.DataFrame({
        'Department Type': [get_fvalue(department_type)],
        'Patient Risk Profile': [get_value(patient_risk_profile,risk_dict)],
        'Minutes to Service': [minutes_to_service],
        'Days in Hospital': [days_in_hospital],
        })

        # Filter the selected one-hot encoded columns
        hospital_cols = [col for col in data.columns if col.startswith('Hospital Branch_')]
        department_cols = [col for col in data.columns if col.startswith('Department_')]
        # Ensure the input data matches the same format as training data
        input_data[hospital_cols] = 0
        input_data[hospital_branch] = 1
        input_data[department_cols] = 0
        input_data[department] = 1
    
        prediction = model.predict(input_data)

        st.subheader("Prediction")
        st.write("Predicted Revenue:", round(prediction[0]))
        st.success("Predicted")

