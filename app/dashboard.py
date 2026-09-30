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

# ---- Store the prediction in session_state so it survives the download button's rerun ----
if predict_clicked:
    input_data = pd.DataFrame([[attendance, test_score, assignments, study_hours, gpa, extracurricular_val]],
                                columns=['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
                                         'study_hours_per_week', 'previous_semester_gpa', 'extracurricular_participation'])
    prediction = model.predict(input_data)[0]
    
    st.session_state['prediction'] = prediction
    st.session_state['student_name'] = student_name
    st.session_state['student_usn'] = student_usn
    st.session_state['attendance'] = attendance
    st.session_state['test_score'] = test_score
    st.session_state['assignments'] = assignments
    st.session_state['study_hours'] = study_hours
    st.session_state['gpa'] = gpa
    st.session_state['extracurricular'] = extracurricular

# ---- MAIN AREA: show results if a prediction exists in session_state ----
if 'prediction' in st.session_state:
    prediction = st.session_state['prediction']
    student_name = st.session_state['student_name']
    student_usn = st.session_state['student_usn']
    attendance = st.session_state['attendance']
    test_score = st.session_state['test_score']
    assignments = st.session_state['assignments']
    study_hours = st.session_state['study_hours']
    gpa = st.session_state['gpa']
    extracurricular = st.session_state['extracurricular']

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

    avg_values = df[['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct']].mean()
    student_values = [attendance, test_score, assignments]
    labels = ['Attendance %', 'Test Score', 'Assignments %']

    # ---- PDF Report with text + chart on ONE page ----
        # ---- PDF Report with text + chart on ONE page ----
    from matplotlib.gridspec import GridSpec

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(2, 1, height_ratios=[1, 1.3], hspace=0.4, top=0.95, bottom=0.05, left=0.1, right=0.9)

        # Top: text details
        text_ax = fig.add_subplot(gs[0])
        text_ax.axis('off')
        text_ax.text(0, 1.0, "Student Academic Risk Report", fontsize=18, fontweight='bold', va='top', transform=text_ax.transAxes)
        details = f"""Student Name: {student_name if student_name else 'N/A'}
USN: {student_usn if student_usn else 'N/A'}

Attendance Percentage: {attendance}%
Average Test Score: {test_score}
Assignments Submitted: {assignments}%
Study Hours per Week: {study_hours}
Previous Semester GPA: {gpa}
Extracurricular Participation: {extracurricular}

Predicted Risk Level: {prediction}

Remark: {remarks[prediction]}
"""
        text_ax.text(0, 0.85, details, fontsize=10.5, va='top', transform=text_ax.transAxes)

        # Bottom: chart
        chart_ax = fig.add_subplot(gs[1])
        x = range(3)
        chart_ax.bar([i - 0.2 for i in x], student_values, width=0.4, label='This Student', color='#3498db')
        chart_ax.bar([i + 0.2 for i in x], avg_values.values, width=0.4, label='Dataset Average', color='#95a5a6')
        chart_ax.set_xticks(x)
        chart_ax.set_xticklabels(labels)
        chart_ax.set_title("Student vs Dataset Average")
        chart_ax.legend()

        pdf.savefig(fig)
        plt.close(fig)