import numpy as np
import matplotlib.pyplot as plt

CATEGORIES = ["Low", "Medium", "High"]
DONUT_COLORS = ['#2ecc71', '#f39c12', '#e74c3c']


def _draw_pair(ax, labels, s_vals, a_vals, ymax, subtitle, show_legend=False):
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


def _two_panel(title, left, right):
    """left / right = (labels, student_values, average_values, ymax, subtitle)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6),
                                   gridspec_kw={'width_ratios': [len(left[0]), len(right[0])]})
    _draw_pair(ax1, *left, show_legend=True)
    _draw_pair(ax2, *right)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def make_academic_chart(vals, avg):
    return _two_panel(
        "Academic Performance Comparison",
        (['Attendance %', 'Test Score', 'Assignments %'],
         [vals['attendance_percentage'], vals['avg_test_score'], vals['assignments_submitted_pct']],
         [avg['attendance_percentage'], avg['avg_test_score'], avg['assignments_submitted_pct']],
         110, "Percentage scale (0-100)"),
        (['GPA'], [vals['previous_semester_gpa']], [avg['previous_semester_gpa']], 11, "GPA scale (0-10)"),
    )


def make_engagement_chart(vals, avg):
    hours_max = max(vals['study_hours_per_week'], vals['library_hours_per_week'],
                    avg['study_hours_per_week'], avg['library_hours_per_week'], 8) * 1.3
    return _two_panel(
        "Study & Engagement Comparison",
        (['Study Hrs/Week', 'Library Hrs/Week'],
         [vals['study_hours_per_week'], vals['library_hours_per_week']],
         [avg['study_hours_per_week'], avg['library_hours_per_week']],
         hours_max, "Hours per week"),
        (['Activities', 'Achievements'],
         [vals['num_activities'], vals['achievements']],
         [avg['num_activities'], avg['achievements']],
         6, "Count (0-5)"),
    )


def make_donut(probs, prediction, confidence):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.pie([probs.get(c, 0) for c in CATEGORIES], colors=DONUT_COLORS, startangle=90,
           counterclock=False, wedgeprops=dict(width=0.38, edgecolor='white'))
    ax.text(0, 0, f"{prediction}\n{confidence:.0f}%", ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.legend(CATEGORIES, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8, frameon=False)
    return fig