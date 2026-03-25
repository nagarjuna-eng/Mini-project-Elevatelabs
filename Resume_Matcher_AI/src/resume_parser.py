import pdfplumber
import os
from src.text_preprocessing import preprocess_text


def extract_text_from_pdf(pdf_path):
    """
    Extract text safely from PDF
    """
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

    except Exception as e:
        print(f"⚠ Error reading {pdf_path}: {e}")
        return None

    return text


def load_resumes(resume_folder):
    """
    Load and preprocess all valid resumes
    """
    resumes = {}

    for file in os.listdir(resume_folder):

        if file.endswith(".pdf"):

            path = os.path.join(resume_folder, file)

            raw_text = extract_text_from_pdf(path)

            if raw_text is None or len(raw_text.strip()) == 0:
                print(f"⚠ Skipping invalid or empty file: {file}")
                continue

            cleaned_text = preprocess_text(raw_text)

            resumes[file] = cleaned_text

    return resumes