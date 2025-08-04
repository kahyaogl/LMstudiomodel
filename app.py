

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from LMmodel import ask_question_from_pdf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "pdf" not in request.files or "question" not in request.form:
            return "PDF veya soru eksik!", 400

        pdf = request.files["pdf"]
        question = request.form["question"]

        filename = secure_filename(pdf.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf.save(file_path)

        answer = ask_question_from_pdf(file_path, question)
        return render_template("index.html", answer=answer)

    return render_template("index.html", answer=None)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
