from flask import Flask, render_template, request
from predict import predict_image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    image_path = ""
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            result = predict_image(image_path)
    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
