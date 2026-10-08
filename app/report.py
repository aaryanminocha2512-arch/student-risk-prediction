import io
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

STATUS_PDF = {'pos': "Positive", 'neu': "Neutral", 'bad': "Needs Attention"}
CATEGORIES = ["Low", "Medium", "High"]


def build_pdf(name, usn, prediction, confidence, conf_text, probs, vals, status,
              extracurricular, recommendations, model_info, charts):
    """charts = list of matplotlib figures to put on the last page."""
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

        def line(label, col, unit=""):
            """One line: label, student's value and the status word."""
            add(f"{label}: {vals[col]:g}{unit}  -  {STATUS_PDF[status[col]]}")

        new_page()
        add("STUDENT ACADEMIC RISK REPORT", size=18, bold=True, gap=0.05)

        heading("1. Student Information")
        add(f"Name: {name}")
        add(f"USN: {usn}")

        heading("2. Prediction")
        add(f"Risk Level: {prediction.upper()}")
        add(f"Prediction Confidence: {confidence:.1f}% ({conf_text} confidence)")

        heading("3. Prediction Probability")
        for c in CATEGORIES:
            add(f"{c} Risk: {probs.get(c, 0)}%")

        heading("4. Academic Performance")
        line("Attendance", 'attendance_percentage', "%")
        line("Test Score", 'avg_test_score')
        line("Assignments Submitted", 'assignments_submitted_pct', "%")
        line("Previous Semester GPA", 'previous_semester_gpa')

        heading("5. Study & Engagement")
        line("Study Hours per Week", 'study_hours_per_week')
        line("Library Hours per Week", 'library_hours_per_week')
        line("Backlogs", 'backlogs')
        add(f"Extracurricular Participation: {extracurricular}")
        line("Activities/Clubs", 'num_activities')
        line("Achievements/Prizes", 'achievements')

        heading("6. Recommended Actions")
        for r in recommendations:
            add("•  " + r, sub="     ")

        heading("7. Model Information")
        add(f"Algorithm: {model_info['algorithm']}")
        add(f"Number of features: {model_info['n_features']}   |   Number of classes: {model_info['n_classes']}")
        add(f"Dataset size: {model_info['dataset_size']} students")
        m = model_info['metrics']
        if m:
            add(f"Training/Test split: {m['n_train']} / {m['n_test']} students")
            add(f"Accuracy: {m['Accuracy']:.2f}   Precision: {m['Precision']:.2f}   "
                f"Recall: {m['Recall']:.2f}   F1 Score: {m['F1 Score']:.2f}")

        # Last page: charts
        new_page()
        add("Comparison Charts", size=14, bold=True, gap=0.04)
        for chart_fig, bottom in zip(charts, (0.62, 0.30)):
            img_buf = io.BytesIO()
            chart_fig.savefig(img_buf, format="png", dpi=150)
            img_buf.seek(0)
            ax = page["fig"].add_axes([0.08, bottom, 0.84, 0.27])
            ax.imshow(plt.imread(img_buf, format="png"))
            ax.axis("off")

        pdf.savefig(page["fig"])
        plt.close(page["fig"])
    buf.seek(0)
    return buf