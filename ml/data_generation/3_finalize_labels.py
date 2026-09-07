import os
import pandas as pd
import json

PREDICTIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cv_labels", "severity_predictions.csv"))
REVIEWS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cv_labels", "severity_reviews.csv"))
FINAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cv_labels", "severity_annotations.csv"))
REPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cv_labels", "final_audit_report.json"))

def generate_final_dataset():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}")
        
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    
    if os.path.exists(REVIEWS_PATH):
        rev_df = pd.read_csv(REVIEWS_PATH)
    else:
        rev_df = pd.DataFrame(columns=["image_path", "human_severity", "rationale"])
        
    # Merge
    merged_df = pred_df.merge(rev_df, on="image_path", how="left")
    
    final_annotations = []
    
    for idx, row in merged_df.iterrows():
        if pd.notnull(row.get('human_severity')):
            final_severity = row['human_severity']
            rationale = row['rationale']
            label_source = "HUMAN_VERIFIED"
        else:
            if row['requires_review']:
                final_severity = None
                rationale = "HUMAN_REVIEW_REQUIRED"
                label_source = "UNRESOLVED"
            else:
                final_severity = row['ai_severity']
                rationale = f"AI assigned based on edge density {row['edge_density']:.4f}, conf {row['ai_confidence']:.4f}, margin {row.get('margin', 0):.4f}"
                label_source = "AI_ONLY"
                
        final_annotations.append({
            "image_path": row['image_path'],
            "filename": row['filename'],
            "defect_type": row['folder_defect_type'],
            "final_severity": final_severity,
            "rationale": rationale,
            "label_source": label_source,
            "requires_review": row['requires_review']
        })
        
    final_df = pd.DataFrame(final_annotations)
    final_df.to_csv(FINAL_PATH, index=False)
    
    # Generate final audit report
    report = {
        "total_images": len(final_df),
        "ai_only_labels": int((final_df['label_source'] == "AI_ONLY").sum()),
        "human_reviewed_labels": int((final_df['label_source'] == "HUMAN_VERIFIED").sum()),
        "unresolved_labels": int((final_df['label_source'] == "UNRESOLVED").sum()),
        "final_severity_distribution": final_df['final_severity'].value_counts().to_dict(),
        "completion_status": "COMPLETE" if (final_df['label_source'] != "UNRESOLVED").all() else "INCOMPLETE"
    }
    
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Final dataset saved to {FINAL_PATH}")
    print(f"Final Audit Report saved to {REPORT_PATH}")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    generate_final_dataset()
