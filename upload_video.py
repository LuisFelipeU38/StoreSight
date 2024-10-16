from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "uploads/videos/"
IMAGES_FOLDER = "uploads/images/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "your_secret_key_here"

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path):
    """Extrae frames del video y los guarda como imágenes."""
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_path = os.path.join(IMAGES_FOLDER, f"frame_{count}.jpg")
        cv2.imwrite(image_path, frame)
        count += 1

    cap.release()
    return count

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Extraer frames del video
            count = extract_frames(file_path)
            flash(f"{count} frames have been extracted.")
            return redirect(url_for("upload_file"))

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
