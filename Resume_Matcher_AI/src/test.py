from resume_parser import load_resumes
from feature_engineering import compute_similarity
from ranker import rank_candidates, print_rankings
from report_generator import generate_report, save_report
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

resume_path = os.path.join(BASE_DIR, "data", "resumes")
jd_path = os.path.join(BASE_DIR, "data", "job_description.txt")

resumes = load_resumes(resume_path)

if len(resumes) == 0:
    print("❌ No valid resumes found.")
    exit()

with open(jd_path, "r") as f:
    job_desc = f.read()

scores = compute_similarity(job_desc, resumes)

ranked = rank_candidates(scores, top_n=3)

print_rankings(ranked)

# Generate report
report_df = generate_report(job_desc, resumes, scores)

print("\n📊 HR REPORT:\n")
print(report_df)

# Save report
save_report(report_df)