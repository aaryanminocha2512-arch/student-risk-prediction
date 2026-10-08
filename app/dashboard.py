import io
import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Student Academic Risk Predictor", layout="wide")

# ------------------------------------------------------------------
# Load the saved model and the dataset (nothing here is changed)
# ------------------------------------------------------------------
model = joblib.load('app/risk_model.pkl')
df = pd.read_csv('data/students_dataset.csv')

# Feature order must match exactly how the model was trained
feature_cols = ['attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
                'study_hours_per_week', 'previous_semester_gpa', 'extracurricular_participation',
                'backlogs', 'library_hours_per_week', 'achievements', 'num_activities']

avg = df[feature_cols].mean()   # dataset averages (calculated from the real data)

CATEGORIES = ["Low", "Medium", "High"]
ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
CARD_BG = {"Low": "#d4f7dc", "Medium": "#fff3cd", "High": "#f8d7da"}
CARD_TEXT = {"Low": "#1e7e34", "Medium": "#856404", "High": "#a71d2a"}
DONUT_COLORS = ['#2ecc71', '#f39c12', '#e74c3c']

# ------------------------------------------------------------------
# Title and description
# ------------------------------------------------------------------
st.title("🎓 Student Academic Risk Prediction")
st.write(
    "An ML-powered system for identifying students who may require academic support "
    "based on academic performance, attendance, study habits, and extracurricular engagement."
)

# ------------------------------------------------------------------
# Sidebar: inputs grouped in expanders (ranges are unchanged)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# When Predict is clicked: run the model and save the result in session_state
# (session_state keeps the result alive when the download button reruns the page)
# ------------------------------------------------------------------
if predict_clicked:
    values = {
        'attendance_percentage': attendance,
        'avg_test_score': test_score,
        'assignments_submitted_pct': assignments,
        'study_hours_per_week': study_hours,
        'previous_semester_gpa': gpa,
        'extracurricular_participation': extracurricular_val,
        'backlogs': backlogs,
        'library_hours_per_week': library_hours,
        'achievements': achievements,
        'num_activities': num_activities,
    }
    input_data = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    probs = {cls: round(float(p) * 100, 1) for cls, p in zip(model.classes_, proba)}

    st.session_state['result'] = {
        'name': student_name,
        'usn': student_usn,
        'prediction': prediction,
        'probs': probs,
        'values': values,
        'extracurricular': extracurricular,
    }

if 'result' not in st.session_state:
    st.info("👈 Enter the student's details in the sidebar and click **Predict Risk Level**.")
    st.stop()

# ------------------------------------------------------------------
# Read the saved result
# ------------------------------------------------------------------
res = st.session_state['result']
name = res['name'] if res['name'] else "N/A"
usn = res['usn'] if res['usn'] else "N/A"
prediction = res['prediction']
probs = res['probs']
vals = res['values']
extracurricular_text = res['extracurricular']
confidence = probs[prediction]


def confidence_label(p):
    if p >= 70:
        return "high"
    elif p >= 50:
        return "moderate"
    return "low (borderline case)"


# ------------------------------------------------------------------
# Performance analysis: student vs dataset average (real values only)
# ------------------------------------------------------------------
ANALYSIS_ITEMS = [
    ('attendance_percentage', 'Attendance (%)', 'main'),
    ('avg_test_score', 'Test Score', 'main'),
    ('assignments_submitted_pct', 'Assignment Submission (%)', 'main'),
    ('previous_semester_gpa', 'Previous Semester GPA', 'main'),
    ('study_hours_per_week', 'Study Hours / Week', 'main'),
    ('backlogs', 'Backlogs', 'backlog'),
    ('library_hours_per_week', 'Library Hours / Week', 'main'),
    ('num_activities', 'Activities / Clubs', 'engage'),
    ('achievements', 'Achievements / Prizes', 'engage'),
]


def get_status(kind, s, a):
    """Returns 'pos', 'neu' or 'bad' by comparing the student's value with the average."""
    if kind == 'backlog':
        if s == 0:
            return 'pos'
        return 'neu' if s <= a else 'bad'
    if kind == 'engage':
        # extra activities/prizes are a bonus, so we never mark them red
        return 'pos' if s >= a else 'neu'
    ratio = s / a if a > 0 else 1
    if ratio >= 1.05:
        return 'pos'
    if ratio >= 0.90:
        return 'neu'
    return 'bad'


def get_comparison(s, a):
    if a == 0:
        return "Above average" if s > 0 else "Around average"
    if s > a * 1.05:
        return "Above average"
    if s < a * 0.95:
        return "Below average"
    return "Around average"


STATUS_SCREEN = {'pos': "🟢 Positive", 'neu': "🟡 Neutral", 'bad': "🔴 Needs Attention"}
STATUS_PDF = {'pos': "Positive", 'neu': "Neutral", 'bad': "Needs Attention"}

status = {}
analysis_rows = []
for col, label, kind in ANALYSIS_ITEMS:
    s, a = vals[col], avg[col]
    status[col] = get_status(kind, s, a)
    analysis_rows.append({
        'col': col, 'factor': label, 'student': f"{s:g}", 'average': f"{a:.1f}",
        'comparison': get_comparison(s, a), 'status': status[col],
    })


# ------------------------------------------------------------------
# Recommendations (built from the student's real values)
# ------------------------------------------------------------------
def get_recommendations(pred):
    recs = []
    if status['attendance_percentage'] == 'bad':
        recs.append(f"Improve attendance: it is {vals['attendance_percentage']:g}% compared with the dataset average of {avg['attendance_percentage']:.1f}%.")
    if status['avg_test_score'] == 'bad':
        recs.append(f"Focus on improving test scores (yours: {vals['avg_test_score']:g}, dataset average: {avg['avg_test_score']:.1f}) and prepare well for upcoming assessments.")
    if status['assignments_submitted_pct'] == 'bad':
        recs.append(f"Submit assignments consistently (yours: {vals['assignments_submitted_pct']:g}%, dataset average: {avg['assignments_submitted_pct']:.1f}%).")
    if status['previous_semester_gpa'] == 'bad':
        recs.append(f"Work on improving GPA (yours: {vals['previous_semester_gpa']:g}, dataset average: {avg['previous_semester_gpa']:.1f}).")
    if status['study_hours_per_week'] == 'bad':
        recs.append(f"Increase weekly study hours (yours: {vals['study_hours_per_week']:g}, dataset average: {avg['study_hours_per_week']:.1f}).")
    if status['backlogs'] == 'bad':
        recs.append(f"Clear pending backlogs (currently {vals['backlogs']:g}).")
    elif pred == "High" and vals['backlogs'] > 0:
        recs.append(f"Clear pending backlogs (currently {vals['backlogs']:g}).")
    if status['library_hours_per_week'] == 'bad':
        recs.append(f"Spend more time on library study (yours: {vals['library_hours_per_week']:g} hrs/week, dataset average: {avg['library_hours_per_week']:.1f}).")

    if pred == "Low":
        base = ["Maintain current attendance and study habits.",
                "Continue submitting assignments consistently.",
                "Maintain academic performance."]
        return base + ["Also keep an eye on this: " + r for r in recs]
    if pred == "Medium":
        if not recs:
            recs.append("Stay consistent in all areas; small improvements can move the student towards Low risk.")
        return recs
    # High risk
    recs.append("Seek academic guidance or mentoring from faculty.")
    return recs


recommendations = get_recommendations(prediction)


# ------------------------------------------------------------------
# Model information (real values only)
# ------------------------------------------------------------------
def get_model_metrics():
    """Evaluates the saved model on the same 80/20 split used in the notebook
    (random_state=42, stratified). Returns None if anything does not match."""
    try:
        X = df[feature_cols]
        y = df['risk_label']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        preds = model.predict(X_test)
        return {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_test, preds, average='weighted', zero_division=0),
            'n_test': len(X_test),
            'n_train': len(df) - len(X_test),
        }
    except Exception:
        return None


metrics = get_model_metrics()
algo_names = {'LogisticRegression': 'Logistic Regression', 'DecisionTreeClassifier': 'Decision Tree',
              'RandomForestClassifier': 'Random Forest'}
algorithm = algo_names.get(type(model).__name__, type(model).__name__)
n_features = len(feature_cols)
n_classes = len(model.classes_)


# ------------------------------------------------------------------
# Chart helpers
# ------------------------------------------------------------------
def draw_pair(ax, labels, s_vals, a_vals, ymax, subtitle, show_legend=False):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, s_vals, w, label='Student', color='#3498db')
    ax.bar(x + w / 2, a_vals, w, label='Dataset Average', color='#95a5a6')
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, s_vals[i] + ymax * 0.02, f"{s_vals[i]:g}", ha='center', fontsize=8)
        ax.text(x[i] + w / 2, a_vals[i] + ymax * 0.02, f"{a_vals[i]:.1f}", ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, ymax)
    ax.set_title(subtitle, fontsize=9)
    if show_legend:
        ax.legend(fontsize=8)


def make_two_panel_chart(title, left, right):
    """left / right = (labels, student_values, average_values, ymax, subtitle)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6),
                                   gridspec_kw={'width_ratios': [len(left[0]), len(right[0])]})
    draw_pair(ax1, *left, show_legend=True)
    draw_pair(ax2, *right)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def v(col):
    return vals[col]


fig_acad = make_two_panel_chart(
    "Academic Performance Comparison",
    (['Attendance %', 'Test Score', 'Assignments %'],
     [v('attendance_percentage'), v('avg_test_score'), v('assignments_submitted_pct')],
     [avg['attendance_percentage'], avg['avg_test_score'], avg['assignments_submitted_pct']],
     110, "Percentage scale (0-100)"),
    (['GPA'], [v('previous_semester_gpa')], [avg['previous_semester_gpa']], 11, "GPA scale (0-10)"),
)

hours_max = max(v('study_hours_per_week'), v('library_hours_per_week'),
                avg['study_hours_per_week'], avg['library_hours_per_week'], 8) * 1.3
fig_eng = make_two_panel_chart(
    "Study & Engagement Comparison",
    (['Study Hrs/Week', 'Library Hrs/Week'],
     [v('study_hours_per_week'), v('library_hours_per_week')],
     [avg['study_hours_per_week'], avg['library_hours_per_week']],
     hours_max, "Hours per week"),
    (['Activities', 'Achievements'],
     [v('num_activities'), v('achievements')],
     [avg['num_activities'], avg['achievements']],
     6, "Count (0-5)"),
)


def make_donut():
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.pie([probs.get(c, 0) for c in CATEGORIES], colors=DONUT_COLORS, startangle=90,
           counterclock=False, wedgeprops=dict(width=0.38, edgecolor='white'))
    ax.text(0, 0, f"{prediction}\n{confidence:.0f}%", ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.legend(CATEGORIES, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8, frameon=False)
    return fig


# ------------------------------------------------------------------
# PDF report (matplotlib only, no extra library)
# ------------------------------------------------------------------
def build_pdf():
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        page = {"fig": None, "y": 0.0}

        def new_page():
            if page["fig"] is not None:
                pdf.savefig(page["fig"])
                plt.close(page["fig"])
            page["fig"] = plt.figure(figsize=(8.27, 11.69))
            page["y"] = 0.95

        def add(text, size=10, bold=False, gap=0.022, indent=0.08, wrap=90, sub=""):
            text = str(text).replace("$", r"\$")
            for line in (textwrap.wrap(text, width=wrap, subsequent_indent=sub) or [""]):
                if page["y"] < 0.06:
                    new_page()
                page["fig"].text(indent, page["y"], line, fontsize=size,
                                 fontweight="bold" if bold else "normal", va="top")
                page["y"] -= gap

        def heading(text):
            page["y"] -= 0.010
            add(text, size=12, bold=True, gap=0.032)

        new_page()
        add("STUDENT ACADEMIC RISK REPORT", size=18, bold=True, gap=0.05)

        heading("1. Student Information")
        add(f"Name: {name}")
        add(f"USN: {usn}")

        heading("2. Prediction")
        add(f"Risk Level: {prediction.upper()}")
        add(f"Prediction Confidence: {confidence:.1f}% ({confidence_label(confidence)} confidence)")

        heading("3. Prediction Probability")
        for c in CATEGORIES:
            add(f"{c} Risk: {probs.get(c, 0)}%")

        heading("4. Academic Performance")
        add(f"Attendance: {v('attendance_percentage'):g}%  (dataset average {avg['attendance_percentage']:.1f}%)")
        add(f"Test Score: {v('avg_test_score'):g}  (dataset average {avg['avg_test_score']:.1f})")
        add(f"Assignments Submitted: {v('assignments_submitted_pct'):g}%  (dataset average {avg['assignments_submitted_pct']:.1f}%)")
        add(f"Previous Semester GPA: {v('previous_semester_gpa'):g}  (dataset average {avg['previous_semester_gpa']:.1f})")

        heading("5. Study & Engagement")
        add(f"Study Hours per Week: {v('study_hours_per_week'):g}  (dataset average {avg['study_hours_per_week']:.1f})")
        add(f"Library Hours per Week: {v('library_hours_per_week'):g}  (dataset average {avg['library_hours_per_week']:.1f})")
        add(f"Backlogs: {v('backlogs'):g}  (dataset average {avg['backlogs']:.1f})")
        add(f"Extracurricular Participation: {extracurricular_text}")
        add(f"Activities/Clubs: {v('num_activities'):g}  (dataset average {avg['num_activities']:.1f})")
        add(f"Achievements/Prizes: {v('achievements'):g}  (dataset average {avg['achievements']:.1f})")

        heading("6. Key Performance Analysis")
        add("(Comparison of the student's values with dataset averages.)", size=9)
        for r in analysis_rows:
            add(f"{r['factor']}: {r['student']} vs average {r['average']} - {r['comparison']} - {STATUS_PDF[r['status']]}",
                sub="    ")

        heading("7. Recommended Actions")
        for r in recommendations:
            add("•  " + r, sub="     ")

        heading("8. Model Information")
        add(f"Algorithm: {algorithm}")
        add(f"Number of features: {n_features}   |   Number of classes: {n_classes}")
        add(f"Dataset size: {len(df)} students")
        if metrics:
            add(f"Training/Test split: {metrics['n_train']} / {metrics['n_test']} students")
            add(f"Accuracy: {metrics['Accuracy']:.2f}   Precision: {metrics['Precision']:.2f}   "
                f"Recall: {metrics['Recall']:.2f}   F1 Score: {metrics['F1 Score']:.2f}")

        # Charts page
        new_page()
        add("Comparison Charts", size=14, bold=True, gap=0.04)
        for chart_fig, bottom in ((fig_acad, 0.62), (fig_eng, 0.30)):
            img_buf = io.BytesIO()
            chart_fig.savefig(img_buf, format="png", dpi=150)
            img_buf.seek(0)
            img = plt.imread(img_buf, format="png")
            chart_ax = page["fig"].add_axes([0.08, bottom, 0.84, 0.27])
            chart_ax.imshow(img)
            chart_ax.axis("off")

        pdf.savefig(page["fig"])
        plt.close(page["fig"])
    buf.seek(0)
    return buf


# ==================================================================
# MAIN PAGE (in the order of the layout)
# ==================================================================

# 1. Student information
st.subheader("Student Information")
c1, c2 = st.columns(2)
c1.write(f"**Name:** {name}")
c2.write(f"**USN:** {usn}")

# 2. Prediction result
st.subheader("Prediction Result")
card = (
    f"<div style='background-color:{CARD_BG[prediction]}; padding:20px; border-radius:10px;'>"
    f"<h2 style='color:{CARD_TEXT[prediction]}; margin:0;'>{ICONS[prediction]} {prediction.upper()} RISK</h2>"
    f"<p style='color:{CARD_TEXT[prediction]}; margin:8px 0 0 0; font-size:18px;'>"
    f"Prediction Confidence: {confidence:.1f}%</p>"
    f"<p style='color:{CARD_TEXT[prediction]}; margin:4px 0 0 0;'>"
    f"Model predicts {prediction} Risk with {confidence_label(confidence)} confidence.</p>"
    f"</div>"
)
st.markdown(card, unsafe_allow_html=True)

# 3. Prediction probability
st.subheader("Prediction Probability")
p1, p2 = st.columns([1, 1.5])
with p1:
    fig_donut = make_donut()
    st.pyplot(fig_donut)
    plt.close(fig_donut)
with p2:
    st.write("")
    for c in CATEGORIES:
        st.write(f"{ICONS[c]} **{c} Risk:** {probs.get(c, 0)}%")
        st.progress(min(max(probs.get(c, 0) / 100, 0.0), 1.0))

# 4. Why this prediction?
st.subheader("🔍 Why This Prediction?")
st.caption(
    "Performance Analysis: this compares the student's values with the dataset averages. "
    "It is a simple comparison, not a technical explanation of the model's internal calculation."
)
table_rows = [{
    "Factor": r['factor'],
    "Student": r['student'],
    "Dataset Average": r['average'],
    "Comparison": r['comparison'],
    "Status": STATUS_SCREEN[r['status']],
} for r in analysis_rows]
st.dataframe(pd.DataFrame(table_rows), hide_index=True)

# 5. Academic performance comparison
st.subheader("📊 Academic Performance Comparison")
st.pyplot(fig_acad)

# 6. Study & engagement comparison
st.subheader("📚 Study & Engagement Comparison")
st.pyplot(fig_eng)

# 7. Recommended actions
st.subheader("💡 Recommended Actions")
for r in recommendations:
    st.markdown("- " + r)

# 8. About the ML model
st.subheader("🤖 About the Machine Learning Model")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", algorithm)
m2.metric("Features", n_features)
m3.metric("Classes", n_classes)
m4.metric("Dataset Size", len(df))
if metrics:
    st.write(f"**Training / Test split:** {metrics['n_train']} / {metrics['n_test']} students")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    k2.metric("Precision", f"{metrics['Precision']:.2f}")
    k3.metric("Recall", f"{metrics['Recall']:.2f}")
    k4.metric("F1 Score", f"{metrics['F1 Score']:.2f}")
    st.caption("Metrics are calculated live from the saved model on the 20% test data (same split as the notebook). "
               "The comparison of all three models is available in the notebook.")

# 9. Download report
st.subheader("📄 Download Report")
st.download_button(
    label="📥 Download Report (PDF)",
    data=build_pdf(),
    file_name=f"risk_report_{usn if usn != 'N/A' else 'student'}.pdf",
    mime="application/pdf",
)

plt.close(fig_acad)
plt.close(fig_eng)