from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO, solutions

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template('index.html')

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
        return redirect(url_for('uploaded_file', filename=file.filename))

def process_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    output_filename = os.path.splitext(filename)[0] + "_processed.avi"
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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    processed_filename = os.path.splitext(filename)[0] + "_processed.avi"
    return send_from_directory(app.config['PROCESSED_FOLDER'], processed_filename)

if __name__ == "__main__":
    app.run(debug=True)
