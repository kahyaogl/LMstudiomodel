from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import os
from LMmodel import ask_question_from_pdf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    pdf_list = os.listdir(UPLOAD_FOLDER)
    return render_template("index.html", pdf_list=pdf_list, answer=None)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return "PDF eksik!", 400

    pdf = request.files["pdf"]
    filename = secure_filename(pdf.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.save(file_path)

    return redirect("/")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    selected_pdf = request.form.get("selected_pdf")

    if not question or not selected_pdf:
        return "Soru veya PDF seçilmedi", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_pdf)
    answer = ask_question_from_pdf(file_path, question)

    pdf_list = os.listdir(UPLOAD_FOLDER)
    return render_template("index.html", answer=answer, pdf_list=pdf_list)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
