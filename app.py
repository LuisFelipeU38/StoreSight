from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, Response
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from ultralytics import YOLO, solutions
import uuid
import subprocess

app = Flask(__name__, static_folder='static')

# Carpetas para cargar videos y almacenar gráficos
UPLOAD_FOLDER = 'uploads/'
PLOTS_FOLDER = 'StoreSight/plots/'

# Crear las carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER
app.secret_key = 'your_secret_key_here'  # Necesario para usar sesiones

# Extensiones de archivo permitidas para subir
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Carga del modelo YOLO
model = YOLO("yolov8n.pt")

def allowed_file(filename):
    """Verifica si la extensión del archivo es permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Página principal."""
    return render_template('index.html')

@app.route('/data')
def data():
    """Página para subir datos."""
    return render_template('data.html')

@app.route('/show_video/<filename>')
def show_video(filename):
    """Muestra el video procesado."""
    return render_template('show_video.html', filename=filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Maneja la carga y procesamiento del archivo de video."""
    if request.method == 'POST':
        # Verifica si hay un archivo en la solicitud
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('upload_file'))
        
        file = request.files['file']

        # Verifica si se seleccionó un archivo
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('upload_file'))

        # Si el archivo es válido, se guarda y procesa
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            processed_video_path = process_video(file.filename)
            scatter_plot_base64 = generate_scatter_plot(os.path.basename(processed_video_path))
            session['scatter_plot'] = scatter_plot_base64  # Guarda el gráfico en la sesión

            flash('The video has been uploaded and processed successfully')

            return redirect(url_for('show_video', filename=os.path.basename(processed_video_path)))
        else:
            flash('Invalid file format. Please upload a video file.')
            return redirect(url_for('upload_file'))
        
    return render_template('data.html')

@app.route('/analytics')
def analytics():
    """Muestra el gráfico generado a partir del video."""
    scatter_plot = session.get('scatter_plot', None)
    if scatter_plot is None:
        return redirect(url_for('home'))
    return render_template('analytics.html', scatter_plot=scatter_plot)

def process_video(filename):
    """Procesa el video utilizando el modelo YOLO y genera un video de salida en formato AVI con códec XVID, luego lo convierte a MP4."""
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        avi_filename = os.path.splitext(filename)[0] + "_processed.avi"
        avi_path = os.path.join('static', avi_filename)
        os.makedirs('static', exist_ok=True)
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

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))

        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=False,
            shape="circle",
            names=model.names,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("End of video or error reading frame.")
                break
            if im0 is not None:
                tracks = model.track(im0, persist=True, show=False)
                im0 = heatmap_obj.generate_heatmap(im0, tracks)
                video_writer.write(im0)
            else:
                print("Skipped an empty frame.")
                
        cap.release()
        video_writer.release()
        print(f"AVI video processing completed: {avi_path}")
        
        # Convert AVI to MP4 using FFmpeg
        mp4_filename = os.path.splitext(filename)[0] + "_processed.mp4"
        mp4_path = os.path.join('static', mp4_filename)
        convert_to_mp4(avi_path, mp4_path)
        
        print(f"MP4 video conversion completed: {mp4_path}")
        
        # Eliminar el archivo AVI
        if os.path.exists(avi_path):
            os.remove(avi_path)
            print(f"AVI file deleted: {avi_path}")
        
        return mp4_path

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        return None


def convert_to_mp4(input_path, output_path):
    """Convierte un archivo de video AVI a MP4 usando FFmpeg."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-strict', 'experimental',
        '-pix_fmt', 'yuv420p',  # Asegura compatibilidad con la web
        output_path
    ]
    subprocess.run(command, check=True)

@app.route('/static/<filename>')
def serve_static_file(filename):
    file_path = os.path.join('static', filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            content = f.read()
        return Response(content, mimetype='video/mp4')
    else:
        return "File not found", 404

def generate_scatter_plot(filename):
    video_path = os.path.join('static', filename)
    cap = cv2.VideoCapture(video_path)

    # Lee un frame del video para el análisis
    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    # Genera el mapa de calor
    heatmap_obj = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA,
        view_img=False,
        shape="circle",
        names=model.names,
    )
    results = model.track(frame, persist=True, show=False)
    
    # Obtén las coordenadas de los objetos detectados
    coordinates = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Cada 'box' tiene un 'xyxy' atributo que contiene las coordenadas
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            coordinates.append((x_center, y_center))

    # Genera el scatter plot
    if coordinates:
        x_coords, y_coords = zip(*coordinates)
    else:
        x_coords = y_coords = []

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, c='red', marker='o', alpha=0.5)
    plt.title('Scatter Plot of Detected Objects')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Archivo temporal
    scatter_plot_filename = f"{uuid.uuid4()}.png"
    scatter_plot_path = os.path.join(app.config['PLOTS_FOLDER'], scatter_plot_filename)  # Cambia la ruta aquí
    plt.savefig(scatter_plot_path)
    plt.close()

    return scatter_plot_filename

if __name__ == "__main__":
    app.run(debug=True)
