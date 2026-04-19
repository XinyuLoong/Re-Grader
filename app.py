import json
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(page_title="FairGrade", layout="wide")
st.title("FairGrade: Grading Consistency Auditor")
st.caption("An AI agent that audits whether grading feedback is consistent with the rubric and the student's answer.")

with open(DATA_DIR / "cases.json", "r", encoding="utf-8") as f:
    cases_list = json.load(f)
cases = {c["case_id"]: c for c in cases_list}

with open(RESULTS_DIR / "predictions.json", "r", encoding="utf-8") as f:
    preds_list = json.load(f)
preds = {p["case_id"]: p for p in preds_list}

with open(RESULTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

# ---- top metrics ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", metrics.get("accuracy", 0))
m2.metric("Precision (Inconsistent)", metrics.get("precision_inconsistent", 0))
m3.metric("Recall (Inconsistent)", metrics.get("recall_inconsistent", 0))
m4.metric("False Positive Rate", metrics.get("false_positive_rate", 0))

st.divider()

# ---- case picker ----
default_case = "case_02" if "case_02" in cases else list(cases.keys())[0]
case_id = st.selectbox("Choose a case", list(cases.keys()), index=list(cases.keys()).index(default_case))
case = cases[case_id]
pred = preds[case_id]

# ---- verdict banner ----
pred_label = pred["prediction"]
if pred_label == "inconsistent":
    st.error(f"Verdict: {pred_label.upper()}  |  Confidence: {pred['confidence']}")
elif pred_label == "consistent":
    st.success(f"Verdict: {pred_label.upper()}  |  Confidence: {pred['confidence']}")
else:
    st.warning(f"Verdict: {pred_label.upper()}  |  Confidence: {pred['confidence']}")

left, right = st.columns(2)

with left:
    st.subheader("Student Answer")
    st.code(case["student_answer"], language="python")
    st.markdown(f"**TA Score:** {case['ta_score']} / {case['max_score']}")
    st.markdown(f"**TA Feedback:** {case['ta_feedback']}")
    st.markdown(f"**Ground Truth Label (for eval):** {case['ground_truth_label']}")

with right:
    st.subheader("Audit Result")
    st.markdown("**Rubric Criteria Used**")
    for item in pred.get("rubric_criteria_used", []):
        st.write(f"- {item}")

    st.markdown("**Answer Evidence**")
    for item in pred.get("answer_evidence", []):
        st.write(f"- {item}")

    st.markdown("**Reasoning Summary**")
    st.write(pred.get("reasoning_summary", ""))

    st.markdown("**Draft Review Note**")
    if pred.get("draft_review_note"):
        st.info(pred["draft_review_note"])
    else:
        st.write("No review note generated.")

st.divider()

st.subheader("Rubric")
st.write(case["rubric"])

st.divider()

st.subheader("What this system does")
st.markdown("""
- Grounds its judgment in the rubric and the student's answer  
- Flags likely grading inconsistencies  
- Can abstain when evidence is ambiguous  
- Drafts a respectful review note only for high-confidence inconsistent cases  
- Never auto-sends complaints
""")