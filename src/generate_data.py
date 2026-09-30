import pandas as pd
import numpy as np

np.random.seed(42)  # ensures same random data every time you run this

NUM_STUDENTS = 500

# Generate base features
student_ids = [f"S{1000 + i}" for i in range(NUM_STUDENTS)]
attendance = np.random.normal(loc=75, scale=15, size=NUM_STUDENTS).clip(30, 100)
avg_test_score = np.random.normal(loc=65, scale=18, size=NUM_STUDENTS).clip(0, 100)
assignments_submitted_pct = np.random.normal(loc=80, scale=15, size=NUM_STUDENTS).clip(0, 100)
study_hours_per_week = np.random.normal(loc=10, scale=5, size=NUM_STUDENTS).clip(0, 40)
previous_gpa = np.random.normal(loc=6.5, scale=1.5, size=NUM_STUDENTS).clip(0, 10)
extracurricular_participation = np.random.choice([0, 1], size=NUM_STUDENTS, p=[0.6, 0.4])

# Combine features into a "risk score" — lower score = higher risk
risk_score = (
    (attendance / 100) * 0.35 +
    (avg_test_score / 100) * 0.30 +
    (assignments_submitted_pct / 100) * 0.15 +
    (previous_gpa / 10) * 0.15 +
    (study_hours_per_week / 40) * 0.05
)

# Add a little randomness so it's not a perfectly predictable formula
risk_score += np.random.normal(0, 0.05, size=NUM_STUDENTS)

# Convert risk score into 3 categories
def label_risk(score):
    if score >= 0.68:
        return "Low"
    elif score >= 0.58:
        return "Medium"
    else:
        return "High"

risk_label = [label_risk(s) for s in risk_score]

# Build the final dataframe
df = pd.DataFrame({
    "student_id": student_ids,
    "attendance_percentage": attendance.round(1),
    "avg_test_score": avg_test_score.round(1),
    "assignments_submitted_pct": assignments_submitted_pct.round(1),
    "study_hours_per_week": study_hours_per_week.round(1),
    "previous_semester_gpa": previous_gpa.round(2),
    "extracurricular_participation": extracurricular_participation,
    "risk_label": risk_label
})

df.to_csv("data/students_dataset.csv", index=False)
print("Dataset generated:", df.shape)
print(df["risk_label"].value_counts())