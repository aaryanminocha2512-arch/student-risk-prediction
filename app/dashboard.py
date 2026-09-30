import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import io
from matplotlib.backends.backend_pdf import PdfPages

# Load the trained model
model = joblib.load('app/risk_model.pkl')

# Load dataset (for comparison averages)
df = pd.read_csv('data/students_dataset.csv')

st.set_page_config(page_title="Student Academic Risk Predictor", layout="centered")

st.title("🎓 Student Academic Risk Prediction")
st.write("Enter a student's details below to predict their academic risk level.")

# ---- SIDEBAR: all inputs go here ----
st.sidebar.header("Student Details")
student_name = st.sidebar.text_input("Student Name")
student_usn = st.sidebar.text_input("USN")

st.sidebar.header("Academic Inputs")
attendance = st.sidebar.slider("Attendance Percentage", 0, 100, 75)
test_score = st.sidebar.slider("Average Test Score", 0, 100, 65)
assignments = st.sidebar.slider("Assignments Submitted (%)", 0, 100, 80)
study_hours = st.sidebar.slider("Study Hours per Week", 0, 40, 10)
gpa = st.sidebar.slider("Previous Semester GPA", 0.0, 10.0, 6.5)
extracurricular = st.sidebar.selectbox("Extracurricular Participation", ["No", "Yes"])
extracurricular_val = 1 if extracurricular == "Yes" else 0

predict_clicked = st.sidebar.button("Predict Risk Level")

# ---- MAIN AREA: results appear here ----
if predict_clicked:
    input_data = pd.DataFrame([[attendance, test_score, assignments, study_hours, gpa, extracurricular_val]],
                                columns=['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
                                         'study_hours_per_week', 'previous_semester_gpa', 'extracurricular_participation'])
    
    prediction = model.predict(input_data)[0]
    
    color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

    if student_name and student_usn:
        st.markdown(f"### Student: **{student_name}** (USN: {student_usn})")

    risk_colors = {"Low": "#d4f7dc", "Medium": "#fff3cd", "High": "#f8d7da"}
    text_colors = {"Low": "#1e7e34", "Medium": "#856404", "High": "#a71d2a"}

    st.markdown(
        f"""
        <div style="background-color:{risk_colors[prediction]}; padding:20px; border-radius:10px;">
            <h2 style="color:{text_colors[prediction]}; margin:0;">
                {color_map[prediction]} Predicted Risk Level: {prediction}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    remarks = {
        "Low": "This student is performing well and shows no immediate signs of academic risk.",
        "Medium": "This student shows some warning signs and may benefit from monitoring or guidance.",
        "High": "This student shows significant risk factors and should be prioritized for academic support."
    }
    st.info(remarks[prediction])

    # ---- Comparison averages (used in both the PDF and the on-screen chart) ----
    avg_values = df[['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct']].mean()
    student_values = [attendance, test_score, assignments]
    labels = ['Attendance %', 'Test Score', 'Assignments %']

    # ---- PDF Report with text + chart ----
        # ---- PDF Report with text + chart ----
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Text details
        fig_text = plt.figure(figsize=(8.5, 11))
        fig_text.text(0.1, 0.93, "Student Academic Risk Report", fontsize=20, fontweight='bold')
        details = f"""Student Name: {student_name if student_name else 'N/A'}
USN: {student_usn if student_usn else 'N/A'}

Attendance Percentage: {attendance}%
Average Test Score: {test_score}
Assignments Submitted: {assignments}%
Study Hours per Week: {study_hours}
Previous Semester GPA: {gpa}
Extracurricular Participation: {extracurricular}

Predicted Risk Level: {prediction}

Remark:
{remarks[prediction]}
"""
        fig_text.text(0.1, 0.55, details, fontsize=12, va='center')
        pdf.savefig(fig_text)
        plt.close(fig_text)

        # Page 2: Comparison chart
        fig_chart, ax = plt.subplots(figsize=(8.5, 5))
        x = range(3)
        ax.bar([i - 0.2 for i in x], student_values, width=0.4, label='This Student', color='#3498db')
        ax.bar([i + 0.2 for i in x], avg_values.values, width=0.4, label='Dataset Average', color='#95a5a6')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Student vs Dataset Average")
        ax.legend()
        pdf.savefig(fig_chart)
        plt.close(fig_chart)

    pdf_buffer.seek(0)

    st.download_button(
        label="📥 Download Report (PDF)",
        data=pdf_buffer,
        file_name=f"risk_report_{student_usn if student_usn else 'student'}.pdf",
        mime="application/pdf"
    )