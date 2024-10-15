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
from ultralytics import YOLO, solutions
import os
import cv2
import matplotlib.pyplot as plt
import uuid
import subprocess

from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder="static")

# Carpetas para cargar videos y almacenar gráficos
UPLOAD_FOLDER = "uploads/"
PLOTS_FOLDER = "StoreSight/plots/"

# Crear las carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOTS_FOLDER"] = PLOTS_FOLDER
app.secret_key = "your_secret_key_here"  # Necesario para usar sesiones

db = SQLAlchemy(app)

# Extensiones de archivo permitidas para subir
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Carga del modelo YOLO (ya entrenado)
model = YOLO("yolov8n.pt")

# Modelo de usuario para la base de datos
# Modelo de usuario para la base de datos
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)


with app.app_context():
    db.create_all()


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

def allowed_file(filename):
    """Verifica si la extensión del archivo es permitida."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    """Maneja la carga del archivo de video."""
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "error")
            return redirect(url_for("upload_file"))

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(url_for("upload_file"))

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            processed_video_path = process_video(file.filename)
            if processed_video_path is None:
                flash("Error processing video", "error")
                return redirect(url_for("upload_file"))

            scatter_plot_base64 = generate_scatter_plot(os.path.basename(processed_video_path))
            if scatter_plot_base64 is None:
                flash("Error generating scatter plot", "error")
                return redirect(url_for("upload_file"))

            session["scatter_plot"] = scatter_plot_base64
            flash("The video has been uploaded and processed successfully")
            return redirect(url_for("show_video", filename=os.path.basename(processed_video_path)))
        else:
            flash("Invalid file format. Please upload a video file.")
            return redirect(url_for("upload_file"))

    return render_template("data.html")

def process_video(filename):
    """Procesa el video utilizando el modelo entrenado."""
    try:
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        mp4_filename = os.path.splitext(filename)[0] + "_processed.mp4"
        mp4_path = os.path.join("static", mp4_filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error reading video file: {video_path}")
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not (w and h and fps):
            print("Error obtaining video properties.")
            cap.release()
            return None

        heatmap = np.zeros((h, w), dtype=np.float32)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("End of video or error reading frame.")
                break

            results = model.track(im0, persist=True, show=False)

            # Procesar resultados
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if box.cls == 0:  # Detección de persona
                        hand_x = int((x1 + x2) // 2)
                        hand_y = int(y2)

                        if 0 <= hand_x < w and 0 <= hand_y < h:
                            heatmap[hand_y:hand_y + 10, hand_x:hand_x + 10] += 1

            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap / (np.max(heatmap) + 1e-5)), cv2.COLORMAP_JET)
            combined_image = cv2.addWeighted(im0, 0.5, heatmap_color, 0.5, 0)
            out.write(combined_image)

        cap.release()
        out.release()
        print(f"Video processing completed: {mp4_path}")

        return mp4_path

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        return None

def generate_scatter_plot(filename):
    video_path = os.path.join("static", filename)
    cap = cv2.VideoCapture(video_path)

    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    plt.figure(figsize=(10, 6))
    plt.title("Sample Scatter Plot")
    plt.scatter(np.random.rand(10), np.random.rand(10))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plot_filename = f"scatter_plot_{uuid.uuid4()}.png"
    plot_path = os.path.join(app.config["PLOTS_FOLDER"], plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return plot_filename

if __name__ == "__main__":
    app.run(debug=True)
