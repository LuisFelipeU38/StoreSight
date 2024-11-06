from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    session,
    flash,
    Response,
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from collections import defaultdict
import os
import cv2
import subprocess


app = Flask(__name__, static_folder="static")

UPLOAD_FOLDER = "uploads/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "your_secret_key_here"  

db = SQLAlchemy(app)

# Extensiones de archivo permitidas para subir
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Variables de configuración
OVERLAP_THRESHOLD = 0.1  # 10% de superposición para contar como una visita

model = YOLO('static/images/store_model.pt')

# Modelo de usuario para la base de datos
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)


with app.app_context():
    db.create_all()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))

        new_user = User(
            username=username, password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid username or password!", "danger")
            return redirect(url_for("login"))

        session["user"] = username
        flash("Logged in successfully!", "success")
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))


@app.context_processor
def inject_user():
    return dict(logged_in="user" in session)


@app.route("/")
def home():
    """Página principal."""
    return render_template("home.html")


@app.route("/data")
def data():
    """Página para subir datos."""
    return render_template("data.html")


@app.route("/show_video/<filename>")
def show_video(filename):
    """Muestra el video procesado."""
    return render_template("show_video.html", filename=filename)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No selected file", "error")
            return redirect(url_for("upload_file"))

        if allowed_file(file.filename):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            processed_video_path = process_video(file.filename)
            if processed_video_path:
                flash("The video has been uploaded and processed successfully")
                return redirect(url_for("show_video", filename=os.path.basename(processed_video_path)))
            else:
                flash("Error processing video", "error")
                return redirect(url_for("upload_file"))
        else:
            flash("Invalid file format. Please upload a video file.", "error")
            return render_template("data.html")
    return render_template("data.html")

def convert_to_mp4(input_path, output_path):
    """Convierte un archivo de video AVI a MP4 usando FFmpeg."""
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-strict",
        "experimental",
        "-pix_fmt",
        "yuv420p",  # Asegura compatibilidad con la web
        output_path,
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")


def process_video(filename):
    """Procesa el video utilizando el modelo YOLO para detectar interacciones con estanterías.
       Genera un video de salida en formato AVI y luego lo convierte a MP4.
    """
    interaction_count = defaultdict(int)
    is_visiting = defaultdict(bool)
    OVERLAP_THRESHOLD = 0.1  # Umbral de superposición para contar como una "visita"

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    avi_filename = os.path.splitext(filename)[0] + "_processed.avi"
    avi_path = os.path.join("static", avi_filename)
    os.makedirs("static", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    model = YOLO('static/images/store_model.pt')

    if not cap.isOpened():
        print(f"Error reading video file: {video_path}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))

    def calculate_intersection_over_union(rect1, rect2):
        x_left = max(rect1[0], rect2[0])
        y_top = max(rect1[1], rect2[1])
        x_right = min(rect1[2], rect2[2])
        y_bottom = min(rect1[3], rect2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

        union_area = rect1_area + rect2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        shelf_rois = {}
        person_rect = None

        for detection in results[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            cls = int(detection.cls[0])

            if cls == 0:
                person_rect = (x1, y1, x2, y2)
            elif 1 <= cls <= 5:
                shelf_label = f"shelf{cls}"
                shelf_rois[shelf_label] = (x1, y1, x2, y2)

        if person_rect:
            for shelf, roi in shelf_rois.items():
                iou = calculate_intersection_over_union(person_rect, roi)
                if iou >= OVERLAP_THRESHOLD and not is_visiting[shelf]:
                    interaction_count[shelf] += 1
                    is_visiting[shelf] = True
                else:
                    is_visiting[shelf] = False

        video_writer.write(annotated_frame)

    cap.release()
    video_writer.release()
    print(f"Video AVI procesado y guardado en: {avi_path}")

    # Convertir AVI a MP4
    mp4_filename = os.path.splitext(filename)[0] + "_processed.mp4"
    mp4_path = os.path.join("static", mp4_filename)
    convert_to_mp4(avi_path, mp4_path)

    # Eliminar el archivo AVI temporal
    if os.path.exists(avi_path):
        os.remove(avi_path)

    session["shelf_visits"] = dict(interaction_count)
    return mp4_path


@app.route("/static/<filename>")
def serve_static_file(filename):
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            content = f.read()
        return Response(content, mimetype="video/mp4")
    else:
        return "File not found", 404


@app.route("/analytics")
def analytics():
    """Muestra las visitas a las estanterías en el análisis del video."""
    shelf_visits = session.get("shelf_visits", None)
    if shelf_visits is None:
        return redirect(url_for("home"))
    return render_template("analytics.html", shelf_visits=shelf_visits)


if __name__ == "__main__":
    app.run(debug=True)
