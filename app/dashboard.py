import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis import analyse, confidence_label, get_recommendations, get_model_metrics
from charts import CATEGORIES, make_donut, make_academic_chart, make_engagement_chart
from report import build_pdf

st.set_page_config(page_title="Student Academic Risk Predictor", layout="wide")

# ---- Load the saved model and the dataset ----
model = joblib.load('app/risk_model.pkl')
df = pd.read_csv('data/students_dataset.csv')

# Feature order must match exactly how the model was trained
feature_cols = ['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
                'study_hours_per_week', 'previous_semester_gpa', 'extracurricular_participation',
                'backlogs', 'library_hours_per_week', 'achievements', 'num_activities']
avg = df[feature_cols].mean()   # dataset averages (calculated from the real data)

ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
CARD_BG = {"Low": "#d4f7dc", "Medium": "#fff3cd", "High": "#f8d7da"}
CARD_TEXT = {"Low": "#1e7e34", "Medium": "#856404", "High": "#a71d2a"}
STATUS_SCREEN = {'pos': "🟢 Positive", 'neu': "🟡 Neutral", 'bad': "🔴 Needs Attention"}

# ---- Title ----
st.title("🎓 Student Academic Risk Prediction")
st.write("An ML-powered system for identifying students who may require academic support "
         "based on academic performance, attendance, study habits, and extracurricular engagement.")

# ---- Sidebar inputs (grouped in expanders) ----
with st.sidebar:
    with st.expander("Student Information", expanded=True):
        student_name = st.text_input("Student Name")
        student_usn = st.text_input("USN")

    with st.expander("Academic Performance", expanded=True):
        attendance = st.slider("Attendance Percentage", 0, 100, 75)
        test_score = st.slider("Average Test Score", 0, 100, 65)
        assignments = st.slider("Assignments Submitted (%)", 0, 100, 80)
        gpa = st.slider("Previous Semester GPA", 0.0, 10.0, 6.5)
        study_hours = st.slider("Study Hours per Week", 0, 40, 10)
        backlogs = st.slider("Number of Backlogs", 0, 5, 0)
        library_hours = st.slider("Library Hours per Week", 0, 20, 6)

    with st.expander("Extracurricular Activities", expanded=True):
        extracurricular = st.selectbox("Extracurricular Participation", ["No", "Yes"])
        extracurricular_val = 1 if extracurricular == "Yes" else 0
        if extracurricular == "Yes":
            num_activities = st.slider("Number of Activities/Clubs", 0, 5, 1)
            achievements = st.slider("Achievements/Prizes Won", 0, 5, 0)
        else:
            num_activities = 0
            achievements = 0

    predict_clicked = st.button("Predict Risk Level")

# ---- Predict and save the result (session_state keeps it alive after the download click) ----
if predict_clicked:
    values = {
        'attendance_percentage': attendance, 'avg_test_score': test_score,
        'assignments_submitted_pct': assignments, 'study_hours_per_week': study_hours,
        'previous_semester_gpa': gpa, 'extracurricular_participation': extracurricular_val,
        'backlogs': backlogs, 'library_hours_per_week': library_hours,
        'achievements': achievements, 'num_activities': num_activities,
    }
    input_data = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    st.session_state['result'] = {
        'name': student_name, 'usn': student_usn, 'prediction': prediction,
        'probs': {cls: round(float(p) * 100, 1) for cls, p in zip(model.classes_, proba)},
        'values': values, 'extracurricular': extracurricular,
    }

if 'result' not in st.session_state:
    st.info("👈 Enter the student's details in the sidebar and click **Predict Risk Level**.")
    st.stop()

# ---- Read the saved result ----
res = st.session_state['result']
name = res['name'] or "N/A"
usn = res['usn'] or "N/A"
prediction = res['prediction']
probs = res['probs']
vals = res['values']
confidence = probs[prediction]
conf_text = confidence_label(confidence)

# ---- Analysis, recommendations, model info, charts ----
status, analysis_rows = analyse(vals, avg)
recommendations = get_recommendations(prediction, vals, avg, status)
model_info = {
    'algorithm': {'LogisticRegression': 'Logistic Regression', 'DecisionTreeClassifier': 'Decision Tree',
                  'RandomForestClassifier': 'Random Forest'}.get(type(model).__name__, type(model).__name__),
    'n_features': len(feature_cols),
    'n_classes': len(model.classes_),
    'dataset_size': len(df),
    'metrics': get_model_metrics(model, df, feature_cols),
}
fig_acad = make_academic_chart(vals, avg)
fig_eng = make_engagement_chart(vals, avg)

# ================= PAGE (in order) =================
st.subheader("Student Information")
c1, c2 = st.columns(2)
c1.write(f"**Name:** {name}")
c2.write(f"**USN:** {usn}")

st.subheader("Prediction Result")
st.markdown(
    f"<div style='background-color:{CARD_BG[prediction]}; padding:20px; border-radius:10px;'>"
    f"<h2 style='color:{CARD_TEXT[prediction]}; margin:0;'>{ICONS[prediction]} {prediction.upper()} RISK</h2>"
    f"<p style='color:{CARD_TEXT[prediction]}; margin:8px 0 0 0; font-size:18px;'>"
    f"Prediction Confidence: {confidence:.1f}%</p>"
    f"<p style='color:{CARD_TEXT[prediction]}; margin:4px 0 0 0;'>"
    f"Model predicts {prediction} Risk with {conf_text} confidence.</p></div>",
    unsafe_allow_html=True)

st.subheader("Prediction Probability")
p1, p2 = st.columns([1, 1.5])
with p1:
    fig_donut = make_donut(probs, prediction, confidence)
    st.pyplot(fig_donut)
    plt.close(fig_donut)
with p2:
    st.write("")
    for c in CATEGORIES:
        st.write(f"{ICONS[c]} **{c} Risk:** {probs.get(c, 0)}%")
        st.progress(min(max(probs.get(c, 0) / 100, 0.0), 1.0))

st.subheader("🔍 Why This Prediction?")
st.caption("Performance Analysis: this compares the student's values with the dataset averages. "
           "It is a simple comparison, not a technical explanation of the model's internal calculation.")
st.dataframe(pd.DataFrame([{
    "Factor": r['factor'], "Student": r['student'], "Dataset Average": r['average'],
    "Comparison": r['comparison'], "Status": STATUS_SCREEN[r['status']],
} for r in analysis_rows]), hide_index=True)

st.subheader("📊 Academic Performance Comparison")
st.pyplot(fig_acad)

st.subheader("📚 Study & Engagement Comparison")
st.pyplot(fig_eng)

st.subheader("💡 Recommended Actions")
for r in recommendations:
    st.markdown("- " + r)

st.subheader("🤖 About the Machine Learning Model")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", model_info['algorithm'])
m2.metric("Features", model_info['n_features'])
m3.metric("Classes", model_info['n_classes'])
m4.metric("Dataset Size", model_info['dataset_size'])
metrics = model_info['metrics']
if metrics:
    st.write(f"**Training / Test split:** {metrics['n_train']} / {metrics['n_test']} students")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    k2.metric("Precision", f"{metrics['Precision']:.2f}")
    k3.metric("Recall", f"{metrics['Recall']:.2f}")
    k4.metric("F1 Score", f"{metrics['F1 Score']:.2f}")
    st.caption("Metrics are calculated from the saved model on the 20% test data (same split as the notebook). "
               "The comparison of all three models is in the notebook.")

st.subheader("📄 Download Report")
pdf_file = build_pdf(name, usn, prediction, confidence, conf_text, probs, vals, status,
                     res['extracurricular'], recommendations, model_info, [fig_acad, fig_eng])
st.download_button("📥 Download Report (PDF)", data=pdf_file,
                   file_name=f"risk_report_{usn if usn != 'N/A' else 'student'}.pdf",
                   mime="application/pdf")

plt.close(fig_acad)
plt.close(fig_eng)