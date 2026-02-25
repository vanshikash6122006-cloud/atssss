from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import docx2txt
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Extract text from resume file
def extract_text(file):
    text = ""

    if file.filename.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    elif file.filename.endswith(".docx"):
        text = docx2txt.process(file)

    return text


@app.route("/analyze", methods=["POST"])
def analyze():

    resume_file = request.files.get("resume")
    job_description = request.form.get("job")

    if not resume_file or not job_description:
        return jsonify({"error": "Missing resume or job description"}), 400

    resume_text = extract_text(resume_file)

    if not resume_text:
        return jsonify({"error": "Could not read resume file"}), 400

    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_description, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2)
    score = float(similarity[0][0]) * 100

    return jsonify({
        "score": round(score, 2),
        "suggestions": "Improve alignment with job-specific skills and keywords."
    })


if __name__ == "__main__":
    app.run(debug=True)