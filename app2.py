import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# --------------------
# App Config
# --------------------
st.set_page_config(page_title="Salary Estimation", layout="wide")
st.title("💼 Salary Estimation Tool")
st.markdown("**Estimate your salary based on your professional profile.**")

# Load model, scaler, and feature order
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
feature_order = joblib.load("features.pkl")

# --------------------
# Sidebar Section
# --------------------
with st.sidebar:
    st.image("C:/Users/misba/Desktop/3135706.png", width=80) 
    st.header("📊 About This Tool")
    st.markdown("""
        This salary estimator uses a **machine learning model** trained  
        on real-world employee data to provide an approximate salary range  
        based on your professional profile.
    """)
    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
        - Enter realistic values for accurate results.  
        - Try changing departments and job titles.  
        - Satisfaction Level: 0 (low) → 1 (high).  
        - Adjust years of experience to explore growth impact.  
    """)
    st.markdown("---")
    st.info("Developed by Misbah")

# --------------------
# Personal Info Section
# --------------------
st.markdown("### 🧍 Personal Information")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=65, value=30)
        Years_at_Company = st.number_input("Years at company", min_value=0, max_value=40, value=3)
    with col2:
        Satisfaction_Level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.7, format="%.2f")
        Average_Monthly_Hours = st.number_input("Average monthly hours", min_value=120, max_value=310, step=1, value=200)

# --------------------
# Work Info Section
# --------------------
st.markdown("### 🏢 Work Information")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        Promotion_Last_5Years = 1 if st.selectbox("Promotion in last 5 years?", ["No", "Yes"]) == "Yes" else 0
        Gender_Male = 1 if st.selectbox("Gender", ["Female", "Male"]) == "Male" else 0
    with col2:
        dept = st.selectbox("Department", ["Finance", "HR", "Marketing", "Sales", "Other"])
        job = st.selectbox("Job Title", ["Analyst", "Engineer", "HR Specialist", "Manager", "Other"])

# One-hot encoding for department
Department_Finance = 1 if dept == "Finance" else 0
Department_HR = 1 if dept == "HR" else 0
Department_Marketing = 1 if dept == "Marketing" else 0
Department_Sales = 1 if dept == "Sales" else 0

# One-hot encoding for job title
Job_Title_Analyst = 1 if job == "Analyst" else 0
Job_Title_Engineer = 1 if job == "Engineer" else 0
Job_Title_HR_Specialist = 1 if job == "HR Specialist" else 0
Job_Title_Manager = 1 if job == "Manager" else 0

# Input dict
input_dict = {
    "Age": Age,
    "Years_at_Company": Years_at_Company,
    "Satisfaction_Level": Satisfaction_Level,
    "Average_Monthly_Hours": Average_Monthly_Hours,
    "Promotion_Last_5Years": Promotion_Last_5Years,
    "Gender_Male": Gender_Male,
    "Department_Finance": Department_Finance,
    "Department_HR": Department_HR,
    "Department_Marketing": Department_Marketing,
    "Department_Sales": Department_Sales,
    "Job_Title_Analyst": Job_Title_Analyst,
    "Job_Title_Engineer": Job_Title_Engineer,
    "Job_Title_HR Specialist": Job_Title_HR_Specialist,
    "Job_Title_Manager": Job_Title_Manager
}

# Create input vector in correct order
x = [input_dict[feat] for feat in feature_order]

# --------------------
# Prediction Section
# --------------------
st.markdown("### 📈 Salary Prediction")
predict_button = st.button("💰 Predict Salary", use_container_width=True)

if predict_button:
    st.balloons()
    x_scaled = scaler.transform([np.array(x)])
    prediction = model.predict(x_scaled)
    st.success(f"**Estimated Salary:** ₹{prediction[0]:,.2f}")

    # Display input profile chart
    df_input = pd.DataFrame({"Feature": feature_order, "Value": x})
    fig = px.bar(df_input, x="Feature", y="Value", color="Feature",
                 title="Your Input Profile", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
 