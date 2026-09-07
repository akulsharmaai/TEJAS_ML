import os
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="TEJAS Severity Reviewer", layout="wide")

PREDICTIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cv_labels", "severity_predictions.csv"))
REVIEWS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cv_labels", "severity_reviews.csv"))

st.title("TEJAS Track Defect - Severity Reviewer")

if not os.path.exists(PREDICTIONS_PATH):
    st.error("Predictions file not found. Please run the automated labeling pipeline first.")
    st.stop()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(PREDICTIONS_PATH)
    return df

df = load_data()

# Load reviews if they exist
if os.path.exists(REVIEWS_PATH):
    reviews_df = pd.read_csv(REVIEWS_PATH)
    reviewed_paths = set(reviews_df['image_path'].tolist())
else:
    reviews_df = pd.DataFrame(columns=["image_path", "human_severity", "rationale"])
    reviewed_paths = set()

# Filter for review needed
to_review_df = df[(df['requires_review'] == True) & (~df['image_path'].isin(reviewed_paths))]

st.sidebar.header("Review Status")
st.sidebar.write(f"Total Images: {len(df)}")
st.sidebar.write(f"Requiring Review: {df['requires_review'].sum()}")
st.sidebar.write(f"Already Reviewed: {len(reviewed_paths)}")
st.sidebar.write(f"Remaining to Review: {len(to_review_df)}")

if len(to_review_df) == 0:
    st.success("All images requiring review have been reviewed!")
    st.stop()

# Present the first one
current_item = to_review_df.iloc[0]

st.subheader(f"Defect Type: {current_item['folder_defect_type']}")
st.write(f"**AI Predicted Severity:** {current_item['ai_severity']} (Confidence: {current_item['ai_confidence']:.2f})")
st.write(f"**Edge Density:** {current_item['edge_density']:.4f}")

col1, col2 = st.columns(2)

with col1:
    try:
        img = Image.open(current_item['image_path'])
        st.image(img, caption=current_item['filename'], use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load image: {e}")

with col2:
    st.write("### Provide Final Label")
    selected_severity = st.selectbox("Select True Severity:", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], index=["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(current_item['ai_severity']))
    rationale = st.text_area("Rationale for review (required):", "")
    
    if st.button("Submit & Next"):
        if not rationale.strip():
            st.error("Please provide a rationale.")
        else:
            new_review = pd.DataFrame([{
                "image_path": current_item['image_path'],
                "human_severity": selected_severity,
                "rationale": rationale.strip()
            }])
            
            updated_reviews = pd.concat([reviews_df, new_review], ignore_index=True)
            updated_reviews.to_csv(REVIEWS_PATH, index=False)
            st.rerun()
