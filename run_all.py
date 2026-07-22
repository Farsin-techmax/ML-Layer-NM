import os
import subprocess
import json

milestones = [20, 30, 40, 50, 60, 70, 80, 90, 100]
md_file = r"C:\Users\panga\.gemini\antigravity-ide\brain\a8c41cdd-c4ee-42d8-a609-306c7773501f\milestone_metrics_report.md"

# Clear old report
with open(md_file, "w", encoding="utf-8") as f:
    f.write("# Milestone Evaluation Report (Leak-Free)\n\nThis report contains the latest evaluation metrics, Confusion Matrices, and SHAP feature importance for each milestone, completely dropping temporal leak features (Service_Num, Vehicle Age, Current Age, Vehicle Lifetime in Months).\n\n")

for m in milestones:
    print(f"--- Training {m}k ---")
    subprocess.run([r".\venv\Scripts\python", "retrain.py", str(m)])
    print(f"--- Plotting {m}k ---")
    subprocess.run([r".\venv\Scripts\python", "generate_eval_plots.py", str(m)])
    
    # Read metrics
    model_dir = f"models/models_alan/{m}k"
    metrics_q1_path = os.path.join(model_dir, f"metrics_{m}k_Q1_Test.json")
    metrics_q2_path = os.path.join(model_dir, f"metrics_{m}k_Q2_Test.json")
    
    if os.path.exists(metrics_q1_path) and os.path.exists(metrics_q2_path):
        with open(metrics_q1_path) as f:
            q1 = json.load(f)
        with open(metrics_q2_path) as f:
            q2 = json.load(f)
            
        md_section = f"""
## {m}k Milestone

### Performance Metrics
| Metric | Q1 Test | Q2 Test |
|--------|---------|---------|
| **Accuracy** | {q1['Accuracy']*100:.2f}% | {q2['Accuracy']*100:.2f}% |
| **ROC-AUC** | {q1['ROC-AUC']:.4f} | {q2['ROC-AUC']:.4f} |
| **F1-Score** | {q1['F1-Score']:.4f} | {q2['F1-Score']:.4f} |
| **Precision** | {q1['Precision']*100:.2f}% | {q2['Precision']*100:.2f}% |
| **Recall** | {q1['Recall']*100:.2f}% | {q2['Recall']*100:.2f}% |
| **Brier Score** | {q1['Brier']:.4f} | {q2['Brier']:.4f} |

### Confusion Matrices
````carousel
![{m}k Q1 Confusion Matrix](C:/Techmax/cwf/code/models/models_alan/{m}k/cm_Q1.png)
<!-- slide -->
![{m}k Q2 Confusion Matrix](C:/Techmax/cwf/code/models/models_alan/{m}k/cm_Q2.png)
````

### Feature Importance (SHAP)
````carousel
![SHAP Feature Importance (Bar)](C:/Techmax/cwf/code/models/models_alan/{m}k/feature_importance.png)
<!-- slide -->
![SHAP Summary Plot](C:/Techmax/cwf/code/models/models_alan/{m}k/shap_summary.png)
````

---
"""
        with open(md_file, "a", encoding="utf-8") as f:
            f.write(md_section)
        print(f"Added {m}k to report.")
    else:
        print(f"Failed to find metrics for {m}k.")
