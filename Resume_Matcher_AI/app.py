import streamlit as st
import os
import tempfile
import pandas as pd

from src.resume_parser import extract_text_from_pdf
from src.text_preprocessing import preprocess_text
from src.feature_engineering import compute_similarity
from src.ranker import rank_candidates
from src.report_generator import generate_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ResumeMatch AI", layout="wide")

# ---------------- CLEAN UI THEME ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Card */
.card {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
}

/* Titles */
.title {
    font-size: 34px;
    font-weight: bold;
    color: white;
}

.subtitle {
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 20px;
}

/* Candidate name */
.candidate {
    font-size: 18px;
    font-weight: bold;
}

/* Score */
.score {
    font-size: 22px;
    font-weight: bold;
}

/* Skill tags */
.tag {
    display: inline-block;
    background-color: #334155;
    padding: 5px 10px;
    margin: 4px;
    border-radius: 8px;
    font-size: 12px;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🚀 ResumeMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Resume Ranking with Explainable AI</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Settings")
top_n = st.sidebar.slider("Top Candidates", 1, 10, 3)

# ---------------- INPUT ----------------
job_description = st.text_area("📌 Job Description", height=180)

uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- PROCESS ----------------
if st.button("🔥 Analyze Candidates"):

    if not job_description or not uploaded_files:
        st.warning("Please provide both inputs.")
        st.stop()

    with st.spinner("Processing resumes..."):

        with tempfile.TemporaryDirectory() as temp_dir:

            resumes = {}

            for file in uploaded_files:

                path = os.path.join(temp_dir, file.name)

                with open(path, "wb") as f:
                    f.write(file.read())

                text = extract_text_from_pdf(path)

                if text:
                    resumes[file.name] = preprocess_text(text)

    job_clean = preprocess_text(job_description)

    scores = compute_similarity(job_clean, resumes)
    ranked = rank_candidates(scores, top_n)

    report_df = generate_report(job_clean, resumes, scores)

    # ---------------- BEST CANDIDATE ----------------
    best_name, best_score = ranked[0]

    st.markdown("## 🥇 Best Match")

    st.markdown(f"""
    <div class="card">
        <div class="candidate">{best_name}</div>
        <div class="score">{best_score:.1f}% Match</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- ALL CANDIDATES ----------------
    st.markdown("## 📊 Candidate Rankings")

    for i, (name, score) in enumerate(ranked, start=1):

        # Get matched skills
        row = report_df[report_df["Candidate"] == name]
        skills = row["Matched Keywords"].values[0].split(", ")

        # Color logic
        if score > 70:
            label = "🔥 Strong Match"
        elif score > 40:
            label = "⚡ Moderate Match"
        else:
            label = "❌ Weak Match"

        st.markdown(f"""
        <div class="card">
            <div class="candidate">#{i} {name}</div>
            <div class="score">{score:.1f}% Match</div>
            <div>{label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Skill tags
        st.markdown("**Matched Skills:**")
        skill_tags = " ".join([f'<span class="tag">{s}</span>' for s in skills if s])
        st.markdown(skill_tags, unsafe_allow_html=True)

        st.progress(score / 100)

    # ---------------- REPORT ----------------
    st.markdown("---")
    st.markdown("## 📄 Detailed HR Report")

    st.dataframe(report_df, use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Full Report",
        csv,
        "resume_report.csv",
        "text/csv"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "© 2026 ResumeMatch AI | Built by Vishnu Shettihalli",
    unsafe_allow_html=True
)