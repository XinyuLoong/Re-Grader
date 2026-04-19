import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free",
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-OpenRouter-Title": "FairGrade",
    },
)


def load_cases() -> List[Dict[str, Any]]:
    with open(DATA_DIR / "cases.json", "r", encoding="utf-8") as f:
        return json.load(f)


def clean_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def audit_case(case: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = """
You are an academic grading consistency auditor.

Your task:
1. Determine whether the TA's score/feedback is consistent with the rubric and the student's answer.
2. Be conservative. If evidence is insufficient, return "abstain".
3. Never assume the TA is wrong unless the student's answer and rubric clearly support that conclusion.
4. Focus on rubric-grounded evidence only.

Return ONLY valid JSON with this exact schema:
{
  "prediction": "consistent" | "inconsistent" | "abstain",
  "confidence": 0.0,
  "rubric_criteria_used": ["..."],
  "answer_evidence": ["..."],
  "reasoning_summary": "...",
  "draft_review_note": "..."
}

Rules:
- "draft_review_note" should be a respectful 2-4 sentence note ONLY if prediction is "inconsistent" and confidence >= 0.75.
- Otherwise "draft_review_note" must be an empty string.
- Keep "reasoning_summary" to 1-3 sentences.
- "answer_evidence" should be concise quotations or paraphrases from the student answer.
- Do not accuse the TA.
- If the evidence is mixed or unclear, choose "abstain".
- Do not infer exact point deductions unless the rubric explicitly specifies point allocation for the missing criterion.
- If the TA correctly identifies a missing requirement but the exact deduction size is ambiguous, prefer "abstain" rather than "inconsistent".
- Reserve "inconsistent" for cases where the feedback clearly contradicts the student's answer or clearly violates an explicit rubric rule.
"""

    user_prompt = f"""
CASE ID: {case["case_id"]}

QUESTION:
{case["question"]}

RUBRIC:
{case["rubric"]}

STUDENT ANSWER:
{case["student_answer"]}

TA SCORE:
{case["ta_score"]}/{case["max_score"]}

TA FEEDBACK:
{case["ta_feedback"]}
"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = response.choices[0].message.content or ""
    cleaned = clean_json_block(raw_text)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "prediction": "abstain",
            "confidence": 0.0,
            "rubric_criteria_used": [],
            "answer_evidence": [],
            "reasoning_summary": "Model response could not be parsed as valid JSON, so the system abstained.",
            "draft_review_note": "",
        }

    result["case_id"] = case["case_id"]
    result["ground_truth_label"] = case["ground_truth_label"]
    result["ta_score"] = case["ta_score"]
    result["max_score"] = case["max_score"]
    return result


def audit_all_cases() -> List[Dict[str, Any]]:
    cases = load_cases()
    outputs = []
    for case in cases:
        print(f"Auditing {case['case_id']} ...")
        outputs.append(audit_case(case))
    return outputs


def main():
    results = audit_all_cases()

    out_path = RESULTS_DIR / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved predictions to: {out_path}")
    for r in results:
        print(
            f"{r['case_id']}: pred={r['prediction']} "
            f"gt={r['ground_truth_label']} conf={r['confidence']}"
        )


if __name__ == "__main__":
    main()