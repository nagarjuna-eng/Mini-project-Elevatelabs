import pandas as pd
import os


def extract_keywords(text):
    words = text.split()
    return set(words)


def generate_report(job_description, resumes_dict, similarity_scores):

    jd_keywords = extract_keywords(job_description)

    report_data = []

    for resume_name, resume_text in resumes_dict.items():

        resume_keywords = extract_keywords(resume_text)

        matched_keywords = jd_keywords.intersection(resume_keywords)

        report_data.append({
            "Candidate": resume_name,
            "Score": similarity_scores.get(resume_name, 0),
            "Matched Keywords": ", ".join(list(matched_keywords)[:10])
        })

    df = pd.DataFrame(report_data)

    df = df.sort_values(by="Score", ascending=False)

    return df


def save_report(df):
    """
    Save report to ROOT outputs folder
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    output_path = os.path.join(BASE_DIR, "outputs")

    os.makedirs(output_path, exist_ok=True)

    file_path = os.path.join(output_path, "hr_report.csv")

    df.to_csv(file_path, index=False)

    print(f"✅ Report saved at: {file_path}")