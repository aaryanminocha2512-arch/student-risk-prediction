import pandas as pd
import numpy as np

np.random.seed(42)

NUM_STUDENTS = 500

# Existing features
student_ids = [f"S{1000 + i}" for i in range(NUM_STUDENTS)]
attendance = np.random.normal(loc=75, scale=15, size=NUM_STUDENTS).clip(30, 100)
avg_test_score = np.random.normal(loc=65, scale=18, size=NUM_STUDENTS).clip(0, 100)
assignments_submitted_pct = np.random.normal(loc=80, scale=15, size=NUM_STUDENTS).clip(0, 100)
study_hours_per_week = np.random.normal(loc=10, scale=5, size=NUM_STUDENTS).clip(0, 40)
previous_gpa = np.random.normal(loc=6.5, scale=1.5, size=NUM_STUDENTS).clip(0, 10)
extracurricular_participation = np.random.choice([0, 1], size=NUM_STUDENTS, p=[0.6, 0.4])

# New features
backlogs = np.random.poisson(lam=0.8, size=NUM_STUDENTS).clip(0, 5)
library_hours_per_week = np.random.normal(loc=6, scale=4, size=NUM_STUDENTS).clip(0, 20)
achievements = np.random.poisson(lam=0.7, size=NUM_STUDENTS).clip(0, 5)
num_activities = np.random.poisson(lam=1.2, size=NUM_STUDENTS).clip(0, 5)

# Weighted risk score (positive factors add, backlogs subtract)
risk_score = (
    (attendance / 100) * 0.25 +
    (avg_test_score / 100) * 0.20 +
    (assignments_submitted_pct / 100) * 0.10 +
    (previous_gpa / 10) * 0.15 +
    (study_hours_per_week / 40) * 0.05 +
    (library_hours_per_week / 20) * 0.05 +
    (achievements / 5) * 0.05 +
    (num_activities / 5) * 0.05 -
    (backlogs / 5) * 0.10
)

# Add small randomness
risk_score += np.random.normal(0, 0.05, size=NUM_STUDENTS)

high_cutoff = np.percentile(risk_score, 10)
low_cutoff = np.percentile(risk_score, 45)

def label_risk(score):
    if score >= low_cutoff:
        return "Low"
    elif score >= high_cutoff:
        return "Medium"
    else:
        return "High"

risk_label = [label_risk(s) for s in risk_score]
df = pd.DataFrame({
    "student_id": student_ids,
    "attendance_percentage": attendance.round(1),
    "avg_test_score": avg_test_score.round(1),
    "assignments_submitted_pct": assignments_submitted_pct.round(1),
    "study_hours_per_week": study_hours_per_week.round(1),
    "previous_semester_gpa": previous_gpa.round(2),
    "extracurricular_participation": extracurricular_participation,
    "backlogs": backlogs,
    "library_hours_per_week": library_hours_per_week.round(1),
    "achievements": achievements,
    "num_activities": num_activities,
    "risk_label": risk_label
})

df.to_csv("data/students_dataset.csv", index=False)
print("Dataset generated:", df.shape)
print(df["risk_label"].value_counts())