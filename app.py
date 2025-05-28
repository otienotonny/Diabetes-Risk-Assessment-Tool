import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json


# Load the model and preprocessor
model = joblib.load('diabetes_risk_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Set page config
st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")

# Title and description
st.title("Diabetes Risk Assessment Tool")
st.write("""
This tool assesses your risk of developing diabetes based on various health factors.
Please fill in your information below to get your risk assessment.
""")

# Create form for user input
with st.form("diabetes_risk_form"):
    st.header("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age", min_value=18, max_value=120, value=30)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        
    with col2:
        smoking_history = st.selectbox("Smoking History", 
                                     ["Never", "Former", "Current", "Unknown"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)
    
    submitted = st.form_submit_button("Assess Risk")

# When form is submitted
if submitted:
    # Prepare input data
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [1 if hypertension == "Yes" else 0],
        'heart_disease': [1 if heart_disease == "Yes" else 0],
        'smoking_history': [smoking_history.lower()],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })
    
    # Make prediction
    try:
        prediction = model.predict_proba(input_data)
        risk_score = prediction[0][1]  # Probability of diabetes
        
        # Display results
        st.subheader("Risk Assessment Results")
        
        # Risk level interpretation
        if risk_score < 0.2:
            risk_level = "Low Risk"
            color = "green"
        elif risk_score < 0.5:
            risk_level = "Moderate Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
        
        # Display risk score with color
        st.metric(label="Diabetes Risk Score", 
                 value=f"{risk_score:.1%}", 
                 delta=risk_level,
                 delta_color="off")
        
        # Progress bar visualization
        st.progress(risk_score)
        
        # Interpretation
        st.write(f"Based on your inputs, your risk of diabetes is classified as **:{color}[{risk_level}]**.")
        
        # Recommendations based on risk level
        st.subheader("Recommendations")
        if risk_level == "Low Risk":
            st.success("""
            - Maintain your healthy lifestyle
            - Continue regular check-ups
            - Monitor your blood sugar levels annually
            """)
        elif risk_level == "Moderate Risk":
            st.warning("""
            - Consider lifestyle modifications (diet, exercise)
            - Monitor your blood sugar levels more frequently
            - Consult with your healthcare provider about prevention strategies
            - Consider losing weight if overweight
            """)
        else:
            st.error("""
            - Please consult with a healthcare provider immediately
            - Significant lifestyle changes are recommended
            - Regular monitoring of blood sugar levels is essential
            - You may need medical intervention
            """)
        
        # Feature importance explanation
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            st.subheader("Key Factors Influencing Your Risk")
            
            # Get feature importances
            importances = model.named_steps['classifier'].feature_importances_
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
            top_features = feature_importance.sort_values('importance', ascending=False).head(3)
            
            for _, row in top_features.iterrows():
                feature_name = row['feature']
                importance = row['importance']
                
                # Simple interpretation of top features
                if "HbA1c_level" in feature_name:
                    st.write(f"🔴 **High HbA1c Level**: Your HbA1c level of {hba1c_level} is a significant factor.")
                elif "blood_glucose_level" in feature_name:
                    st.write(f"🔴 **Blood Glucose Level**: Your blood glucose level of {blood_glucose_level} impacts your risk.")
                elif "bmi" in feature_name:
                    st.write(f"🔴 **BMI**: Your BMI of {bmi} contributes to your risk assessment.")
                elif "age" in feature_name:
                    st.write(f"🔴 **Age**: At {age} years, age is a contributing factor.")
                elif "hypertension" in feature_name and hypertension == "Yes":
                    st.write("🔴 **Hypertension**: Having hypertension increases your risk.")
                elif "heart_disease" in feature_name and heart_disease == "Yes":
                    st.write("🔴 **Heart Disease**: Existing heart disease is a risk factor.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some additional information
st.sidebar.header("About This Tool")
st.sidebar.write("""
This diabetes risk assessment tool uses machine learning to estimate your probability 
of having diabetes based on key health indicators.

**Note**: This tool is for informational purposes only and is not a substitute 
for professional medical advice. Always consult with a healthcare provider for 
personalized medical advice.
""")

st.sidebar.header("Risk Score Interpretation")
st.sidebar.write("""
- **<20%**: Low Risk  
- **20-50%**: Moderate Risk  
- **>50%**: High Risk  
""")

st.sidebar.header("Normal Ranges")
st.sidebar.write("""
- **HbA1c**: <5.7% (normal), 5.7-6.4% (prediabetes), ≥6.5% (diabetes)  
- **Fasting Blood Glucose**: <100 mg/dL (normal), 100-125 mg/dL (prediabetes), ≥126 mg/dL (diabetes)  
- **BMI**: 18.5-24.9 (normal), 25-29.9 (overweight), ≥30 (obese)  
""")
