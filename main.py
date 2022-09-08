
import streamlit as st
import pandas as pd
from sklearn import linear_model, model_selection

# Containers
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache
def get_csv_data(filename):
    data = pd.read_csv(filename)

    
with header:
    st.title("C-964: Will you have a heart attack?")

with dataset:
    st.title("This is my dataset")
    st.text("I found this dataset on kaggle.com at:\nhttps://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")

    heart_attack_data = pd.read_csv("heart_cleaned.csv")
    #ohe = OneHotEncoder(handle_unknown="ignore")
    
    st.write(heart_attack_data.head(5))
    st.caption("This is the head of my dataset.")


    
with features:
    st.title("These are my descriptives.")

    st.subheader("Is age a good indicator of a cardiac event?")
    age_distribution = pd.DataFrame(heart_attack_data['age'].value_counts())
    st.bar_chart(heart_attack_data, x="age", y="output")
    st.caption("This graph show the number of patients at each age who had a cardiac event.  Where the Y-Axis is the number of patients and the X-Axis is the age of those patients.")

    st.subheader("Is cholesterol level a good indicator of a cardiac event?")
    cholesterol_distribution = pd.DataFrame(heart_attack_data['cholesterol'].value_counts())
    st.bar_chart(heart_attack_data, x = "cholesterol", y="output")
    st.caption("This graph is showing the number of patients with a specific cholesterol, who were at high risk of a cardiac event.  Where the output is the number of patients and chol is the cholesterol")

    st.subheader("Is resting blood pressure a good indicator of a cardiac event?")
    st.bar_chart(heart_attack_data, x = "bloodPressure", y="output")
    st.caption("This graph is showing the number of patients with a specific blood pressure, who were at high risk of a cardiac event.  Where the output is the number of patients and bloodPressure is Blood Pressure.")


with model_training:
    st.title("Enter your patients vitals.")

    select_col, display_col = st.columns(2, gap="large")
    
    # User input
    age_slider = select_col.slider("How old are you?")

    male_or_female = select_col.selectbox("Is the patient male of female?", options=["Male", "Female"])
    if male_or_female == "Male":
        male_or_female = 1
    else:
        male_or_female = 0
    
    type_chest_pain = select_col.selectbox("What type of chest pain is the patient having?", options=["Typical Angina", "Atypical Angina", "Non-Anginal pain", "Asymptomatic"])
    if type_chest_pain == "Typical Angina": # Not converting to int value
        type_chest_pain = 10
    elif type_chest_pain == "Atypical Angina":
        type_chest_pain = 20
    elif type_chest_pain == "Non-Anginal pain":
        type_chest_pain = 30
    elif type_chest_pain == "Asymptomatic":
        type_chest_pain = 0

    resting_systolic_blood_pressure = select_col.slider("What is the patient's resting systolic blood pressure in mm/Hg?", min_value=0, max_value=300, value=120, step=1)

    cholestoral = select_col.slider("What is the patient's cholestoral in mg/dl fetched via BMI sensor?", min_value=0, max_value=500, value=200, step=10) 

    fasting_blood_sugar = select_col.selectbox("Is the patient's fasting blood sugar greater than 120 mg/dl?", options=["Yes", "No"])
    if fasting_blood_sugar == "Yes":
        fasting_blood_sugar = 10
    else:
        fasting_blood_sugar = 0

    resting_electrocardiographic_results = select_col.selectbox("What are the results of the patient's EKG?", options=["Normal", "ST-T wave abnormality greater than 0.05 mV", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
    if resting_electrocardiographic_results == "Normal":
        resting_electrocardiographic_results = 0
    elif resting_electrocardiographic_results == "ST-T wave abnormality greater than 0.05 mV": # Not converting to int value
        resting_electrocardiographic_results = 20
    elif resting_electrocardiographic_results == "Showing probable or definite left ventricular hypertrophy by Estes' criteria":
        resting_electrocardiographic_results = 10

    maximum_heart_rate_achieved = select_col.slider("What was the patient's maximum heart rate?", min_value=0, max_value=300, value=100, step=1)

    exercise_induced_angina = select_col.selectbox("Does the patient have exercise induced angina?", options=["Yes", "No"])
    if exercise_induced_angina == "Yes":
        exercise_induced_angina = 10
    else:
        exercise_induced_angina = 0

    # Logistic Regression Model
    mylog_model = linear_model.LogisticRegression()
    y = heart_attack_data.values[:, 9]
    X = heart_attack_data.values[:, 0 : 9]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.3)

    mylog_model.fit(X_train, y_train)

    input_data = [[
        age_slider, 
        male_or_female,
        type_chest_pain,
        resting_systolic_blood_pressure, 
        cholestoral, 
        fasting_blood_sugar,
        resting_electrocardiographic_results, 
        maximum_heart_rate_achieved,
        exercise_induced_angina        
        ]]

    predication = mylog_model.predict(input_data)
    if predication == 1:
        select_col.header("Your patient is at high risk of a cardiac event!")
    else:
        select_col.header("Your patient is at a low risk of a cardiac event.")

