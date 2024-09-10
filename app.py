from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash
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

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

model = YOLO("yolov8n.pt")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('data.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/show_video/<filename>')
def show_video(filename):
    return render_template('show_video.html', filename=filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('upload_file'))
        
        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('upload_file'))

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            process_video(file.filename)
            processed_filename = os.path.splitext(file.filename)[0] + "_processed.mp4"

            scatter_plot_base64 = generate_scatter_plot(processed_filename)
            session['scatter_plot'] = scatter_plot_base64  # Store scatter plot in session

            flash('The video has been uploaded and processed successfully')
            print("Scatter Plot Stored in Session:", scatter_plot_base64[:100])

            return redirect(url_for('show_video', filename=processed_filename))
        else:
            flash('Invalid file format. Please upload a video file.')
            return redirect(url_for('upload_file'))
        
    return render_template('data.html')

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
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = os.path.splitext(filename)[0] + "_processed.mp4"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

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
        print(f"Video processing completed: {output_path}")
        return output_path

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        return None

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

    # archivo temporal
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
