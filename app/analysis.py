from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# (column name, label shown on screen, type of factor)
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
        # extra activities/prizes are a bonus, so they are never marked red
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


def analyse(vals, avg):
    """Returns (status dictionary, list of rows for the analysis table)."""
    status = {}
    rows = []
    for col, label, kind in ANALYSIS_ITEMS:
        s, a = vals[col], avg[col]
        status[col] = get_status(kind, s, a)
        rows.append({'factor': label, 'student': f"{s:g}", 'average': f"{a:.1f}",
                     'comparison': get_comparison(s, a), 'status': status[col]})
    return status, rows


def confidence_label(p):
    if p >= 70:
        return "high"
    if p >= 50:
        return "moderate"
    return "low (borderline case)"


def get_recommendations(pred, vals, avg, status):
    """Recommendations are built from the student's real values."""
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
    if status['backlogs'] == 'bad' or (pred == "High" and vals['backlogs'] > 0):
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
    recs.append("Seek academic guidance or mentoring from faculty.")   # High risk
    return recs


def get_model_metrics(model, df, feature_cols):
    """Evaluates the saved model on the same 80/20 split used in the notebook.
    Returns None if anything does not match."""
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