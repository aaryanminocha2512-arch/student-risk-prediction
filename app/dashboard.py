import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import joblib
import io
from matplotlib.backends.backend_pdf import PdfPages

# Load the trained model
model = joblib.load('app/risk_model.pkl')

# Load dataset (for comparison averages)
df = pd.read_csv('data/students_dataset.csv')

st.set_page_config(page_title="Student Academic Risk Predictor", layout="wide")

st.title("🎓 Student Academic Risk Prediction")
st.write(
    "This app predicts a student's academic risk level — Low, Medium, or High — "
    "based on their attendance, academic performance, and extracurricular engagement."
)

# ---- SIDEBAR: grouped inputs ----
st.sidebar.header("Student Information")
student_name = st.sidebar.text_input("Student Name")
student_usn = st.sidebar.text_input("USN")

st.sidebar.header("Academic Performance")
attendance = st.sidebar.slider("Attendance Percentage", 0, 100, 75)
test_score = st.sidebar.slider("Average Test Score", 0, 100, 65)
assignments = st.sidebar.slider("Assignments Submitted (%)", 0, 100, 80)
gpa = st.sidebar.slider("Previous Semester GPA", 0.0, 10.0, 6.5)
study_hours = st.sidebar.slider("Study Hours per Week", 0, 40, 10)
backlogs = st.sidebar.slider("Number of Backlogs", 0, 5, 0)
library_hours = st.sidebar.slider("Library Hours per Week", 0, 20, 6)

st.sidebar.header("Extracurricular Activities")
extracurricular = st.sidebar.selectbox("Extracurricular Participation", ["No", "Yes"])
extracurricular_val = 1 if extracurricular == "Yes" else 0

if extracurricular == "Yes":
    num_activities = st.sidebar.slider("Number of Activities/Clubs", 0, 5, 1)
    achievements = st.sidebar.slider("Achievements/Prizes Won", 0, 5, 0)
else:
    num_activities = 0
    achievements = 0

predict_clicked = st.sidebar.button("Predict Risk Level")

# Feature order must exactly match how the model was trained
feature_cols = ['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
                 'study_hours_per_week', 'previous_semester_gpa', 'extracurricular_participation',
                 'backlogs', 'library_hours_per_week', 'achievements', 'num_activities']

# ---- Store prediction in session_state so it survives the download button's rerun ----
if predict_clicked:
    input_values = [attendance, test_score, assignments, study_hours, gpa, extracurricular_val,
                     backlogs, library_hours, achievements, num_activities]
    input_data = pd.DataFrame([input_values], columns=feature_cols)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    class_order = model.classes_
    risk_pct = dict(zip(class_order, (probabilities * 100).round(1)))

    st.session_state['prediction'] = prediction
    st.session_state['risk_pct'] = risk_pct
    st.session_state['student_name'] = student_name
    st.session_state['student_usn'] = student_usn
    st.session_state['input_values'] = input_values
    st.session_state['extracurricular'] = extracurricular

# ---- MAIN AREA ----
if 'prediction' in st.session_state:
    prediction = st.session_state['prediction']
    risk_pct = st.session_state['risk_pct']
    student_name = st.session_state['student_name']
    student_usn = st.session_state['student_usn']
    (attendance, test_score, assignments, study_hours, gpa, extracurricular_val,
     backlogs, library_hours, achievements, num_activities) = st.session_state['input_values']
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
                {color_map[prediction]} Predicted Risk Level: {prediction} ({risk_pct[prediction]}% confidence)
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("**Risk breakdown across all categories:**")
    st.write(f"🟢 Low: {risk_pct.get('Low', 0)}%  |  🟡 Medium: {risk_pct.get('Medium', 0)}%  |  🔴 High: {risk_pct.get('High', 0)}%")

    # ---- Identify which factors are driving the risk (compare to dataset averages) ----
    avg = df[feature_cols].mean()
    student_vals = dict(zip(feature_cols, [attendance, test_score, assignments, study_hours, gpa,
                                            extracurricular_val, backlogs, library_hours, achievements, num_activities]))

    # higher-is-better features vs lower-is-better (backlogs)
    higher_is_better = ['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
                         'study_hours_per_week', 'previous_semester_gpa', 'library_hours_per_week',
                         'achievements', 'num_activities']
    weak_areas = []
    for col in higher_is_better:
        if student_vals[col] < avg[col] * 0.85:  # notably below average
            weak_areas.append(col)
    if student_vals['backlogs'] > avg['backlogs'] + 0.5:
        weak_areas.append('backlogs')

    readable_names = {
        'attendance_percentage': 'Attendance',
        'avg_test_score': 'Test Scores',
        'assignments_submitted_pct': 'Assignment Submission',
        'study_hours_per_week': 'Study Hours',
        'previous_semester_gpa': 'GPA',
        'library_hours_per_week': 'Library Usage',
        'achievements': 'Achievements',
        'num_activities': 'Extracurricular Involvement',
        'backlogs': 'Backlogs'
    }
    weak_areas_readable = [readable_names[w] for w in weak_areas]

    remarks = {
        "Low": "This student is performing well and shows no immediate signs of academic risk.",
        "Medium": "This student shows some warning signs and may benefit from monitoring or guidance.",
        "High": "This student shows significant risk factors and should be prioritized for academic support."
    }

    if prediction != "Low" and weak_areas_readable:
        improvement_text = (
            f"To reduce risk, this student should focus on improving: **{', '.join(weak_areas_readable)}**."
        )
    elif prediction != "Low":
        improvement_text = "This student is borderline — maintaining consistency across all areas is recommended."
    else:
        improvement_text = ""

    st.info(remarks[prediction] + (" " + improvement_text if improvement_text else ""))

    # ---- Comparison chart: student vs average across all features ----
    chart_features = ['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
                       'study_hours_per_week', 'library_hours_per_week', 'achievements', 'num_activities']
    chart_labels = ['Attendance %', 'Test Score', 'Assignments %', 'Study Hrs/Wk',
                     'Library Hrs/Wk', 'Achievements', 'Activities']
    student_chart_vals = [student_vals[c] for c in chart_features]
    avg_chart_vals = [avg[c] for c in chart_features]

    def draw_comparison_chart(ax):
        x = np.arange(len(chart_labels))
        width = 0.35
        ax.bar(x - width/2, student_chart_vals, width, label='This Student', color='#3498db')
        ax.bar(x + width/2, avg_chart_vals, width, label='Dataset Average', color='#95a5a6')
        ax.set_xticks(x)
        ax.set_xticklabels(chart_labels, rotation=30, ha='right')
        ax.set_title("Student vs Dataset Average (All Factors)")
        ax.legend()

    # ---- PDF Report ----
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(2, 1, height_ratios=[1, 1.3], hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.9)

        text_ax = fig.add_subplot(gs[0])
        text_ax.axis('off')
        text_ax.text(0, 1.0, "Student Academic Risk Report", fontsize=18, fontweight='bold', va='top', transform=text_ax.transAxes)
        details = f"""Student Name: {student_name if student_name else 'N/A'}
USN: {student_usn if student_usn else 'N/A'}

Attendance: {attendance}%   Test Score: {test_score}   Assignments: {assignments}%
GPA: {gpa}   Study Hours/Week: {study_hours}   Backlogs: {backlogs}
Library Hours/Week: {library_hours}   Achievements: {achievements}   Activities: {num_activities}
Extracurricular Participation: {extracurricular}

Predicted Risk Level: {prediction} ({risk_pct[prediction]}% confidence)
Risk Breakdown: Low {risk_pct.get('Low',0)}% | Medium {risk_pct.get('Medium',0)}% | High {risk_pct.get('High',0)}%

Remark: {remarks[prediction]}
{improvement_text}
"""
        text_ax.text(0, 0.85, details, fontsize=10, va='top', transform=text_ax.transAxes)

        chart_ax = fig.add_subplot(gs[1])
        draw_comparison_chart(chart_ax)

        pdf.savefig(fig)
        plt.close(fig)

    pdf_buffer.seek(0)

    st.download_button(
        label="📥 Download Report (PDF)",
        data=pdf_buffer,
        file_name=f"risk_report_{student_usn if student_usn else 'student'}.pdf",
        mime="application/pdf"
    )

    # ---- On-screen comparison chart ----
    st.subheader("How this student compares to dataset averages")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    draw_comparison_chart(ax2)
    st.pyplot(fig2)