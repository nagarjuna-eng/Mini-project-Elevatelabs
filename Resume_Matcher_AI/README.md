# 🚀 ResumeMatch AI – Intelligent Resume Ranking System

## By Pranav SP

ResumeMatch AI is a smart NLP-powered system that automatically ranks resumes based on their relevance to a given job description. It helps recruiters quickly identify the best candidates using data-driven insights.

---

## 🌐 Live Demo

👉 https://resume-match-ai-1.streamlit.app/

---

## 📌 Overview

This project uses Natural Language Processing (NLP) and Machine Learning techniques to:

- Analyze job descriptions
- Extract meaningful features from resumes
- Rank candidates based on similarity
- Provide explainable insights for HR decision-making

---

## 💡 Key Features

- 📊 Resume ranking based on job description
- 🧠 NLP preprocessing using TF-IDF
- 🎯 Match score displayed as percentage
- 🏷 Skill highlighting (matched keywords)
- 📄 Downloadable HR report
- ⚡ Fast and interactive Streamlit UI

---

## 🧠 How It Works

1. Job description is preprocessed
2. Resumes are parsed from PDF files
3. Text is cleaned and transformed using TF-IDF
4. Cosine similarity is computed
5. Candidates are ranked based on relevance
6. Results are visualized with scores and insights

---

## 🏗 Project Structure
```
ResumeMatch-AI/
│
├── app.py                     # Streamlit application (main UI)
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── report.pdf                 # Internship / project report
│
├── data/
│   ├── job_description.txt    # Sample job description
│   └── resumes/              # Sample resume PDFs
│
├── models/                   # (Optional) Saved models (if used)
│
├── outputs/
│   ├── hr_report.csv         # Generated HR ranking report
│   └── screenshots/          # App screenshots for README
│
└── src/
    ├── __init__.py
    ├── resume_parser.py      # Extract text from PDFs
    ├── text_preprocessing.py # Clean & preprocess text
    ├── feature_engineering.py# TF-IDF + similarity logic
    ├── ranker.py             # Ranking logic
    ├── report_generator.py   # Generate HR report
    └── test.py               # Local testing script
```


---

## ⚙ Tech Stack

- Python
- Scikit-learn (TF-IDF, cosine similarity)
- NLTK (text preprocessing)
- PDFPlumber (PDF parsing)
- Streamlit (web app deployment)

---

## 📊 Sample Output

- Candidate ranking with percentage match
- Skill-based insights
- Visual progress indicators
- HR-friendly report table

---

## 📸 Demo

Screenshots are in the outputs folder showcasing a demo on how the ResumeMatch AI works.

---

## 🚀 Deployment

The application is deployed using Streamlit Cloud and is accessible via the live demo link above.

---

## 🧠 Key Learnings

- Built end-to-end NLP pipeline
- Implemented real-world ranking system
- Improved usability with percentage scoring
- Designed UI for non-technical users
- Handled deployment challenges and dependency management

---
---

## ⭐ If you found this useful, consider giving it a star!
