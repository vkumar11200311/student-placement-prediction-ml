import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)


# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("placement_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model()


# Title
st.title("🎓 Student Placement Prediction")

st.write(
    "Enter the student's academic and skill information "
    "to predict their placement probability."
)

st.info(
    "This is an educational machine learning project "
    "based on a synthetic dataset."
)


# Student information
st.subheader("📋 Student Information")


# IQ
iq = st.number_input(
    "IQ Score",
    min_value=70,
    max_value=145,
    value=100,
    step=1
)


# CGPA
cgpa = st.number_input(
    "CGPA",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.01
)


# 10th Marks
tenth_marks = st.number_input(
    "10th Marks (%)",
    min_value=40.0,
    max_value=100.0,
    value=70.0,
    step=0.1
)


# 12th Marks
twelfth_marks = st.number_input(
    "12th Marks (%)",
    min_value=40.0,
    max_value=100.0,
    value=70.0,
    step=0.1
)


# Technical Skills
technical_skills = st.slider(
    "Technical Skills (1-10)",
    min_value=1.0,
    max_value=10.0,
    value=6.0,
    step=0.1
)


# Communication Skills
communication_skills = st.slider(
    "Communication Skills (1-10)",
    min_value=1.0,
    max_value=10.0,
    value=6.0,
    step=0.1
)


# Prediction button
if st.button("🔮 Predict Placement", use_container_width=True):

    # Create DataFrame
    new_student = pd.DataFrame(
        [[
            iq,
            cgpa,
            tenth_marks,
            twelfth_marks,
            technical_skills,
            communication_skills
        ]],
        columns=[
            "IQ",
            "CGPA",
            "10th_Marks",
            "12th_Marks",
            "Technical_Skills",
            "Communication_Skills"
        ]
    )


    # Scale input
    new_student_scaled = scaler.transform(
        new_student
    )


    # Make prediction
    prediction = model.predict(
        new_student_scaled
    )[0]


    # Calculate probability
    probability = model.predict_proba(
        new_student_scaled
    )[0][1] * 100


    # Display result
    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.success(
            "🎉 Student is likely to be PLACED"
        )

    else:

        st.error(
            "⚠️ Student is likely to be NOT PLACED"
        )


    # Probability
    st.metric(
        "Placement Probability",
        f"{probability:.2f}%"
    )


    # Progress bar
    st.progress(
        int(probability)
    )


    # Student summary
    st.subheader("📋 Student Input Summary")

    summary = pd.DataFrame({
        "Feature": [
            "IQ",
            "CGPA",
            "10th Marks",
            "12th Marks",
            "Technical Skills",
            "Communication Skills"
        ],
        "Value": [
            iq,
            cgpa,
            f"{tenth_marks}%",
            f"{twelfth_marks}%",
            f"{technical_skills}/10",
            f"{communication_skills}/10"
        ]
    })

    st.table(summary)


# Footer
st.divider()

st.caption(
    "Student Placement Prediction System | "
    "Machine Learning Project | Python + Streamlit"
)