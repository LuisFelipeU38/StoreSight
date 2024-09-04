from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from ultralytics import YOLO, solutions
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
PLOTS_FOLDER = 'plots/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER
app.secret_key = 'your_secret_key_here'  # Necesario para usar sesiones

model = YOLO("yolov8n.pt")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        process_video(file.filename)
        processed_filename = os.path.splitext(file.filename)[0] + "_processed.mp4"
        scatter_plot_base64 = generate_scatter_plot(processed_filename)
        session['scatter_plot'] = scatter_plot_base64  # Store scatter plot in session
        print("Scatter Plot Stored in Session:", scatter_plot_base64[:100])
        return redirect(url_for('show_video', filename=processed_filename))

@app.route('/show_video/<filename>')
def show_video(filename):
    return render_template('show_video.html', filename=filename)

@app.route('/analytics')
def analytics():
    scatter_plot = session.get('scatter_plot', None)
    if scatter_plot is None:
        return redirect(url_for('home'))
    return render_template('analytics.html', scatter_plot=scatter_plot)

@app.route('/processed/<filename>')
def serve_processed_file(filename):
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    except FileNotFoundError:
        return "File not found", 404
    
def process_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    output_filename = os.path.splitext(filename)[0] + "_processed.mp4"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    heatmap_obj = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA,
        view_img=False,
        shape="circle",
        names=model.names,
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        tracks = model.track(im0, persist=True, show=False)
        im0 = heatmap_obj.generate_heatmap(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()

def generate_scatter_plot(filename):
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
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

    """ Guarda la figura en un buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    scatter_plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()"""

    # Guarda la figura en un archivo temporal
    scatter_plot_filename = f"{uuid.uuid4()}.png"
    scatter_plot_path = os.path.join(app.config['PLOTS_FOLDER'], scatter_plot_filename)
    plt.savefig(scatter_plot_path)
    plt.close()

    return scatter_plot_filename

@app.route('/plots/<filename>')
def serve_plot_file(filename):
    return send_from_directory(app.config['PLOTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
