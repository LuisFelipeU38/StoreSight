import os
from ultralytics import YOLO

# Carpetas de datos y etiquetas
DATA_FOLDER = "uploads/extracted_images/"  # Carpeta que contiene las imágenes
LABELS_FOLDER = "uploads/extracted_images/annotations/"  # Carpeta que contendrá las etiquetas

# Crear las carpetas si no existen
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

# Archivo de configuración YOLO para tus datos
CONFIG_FILE = "custom_config.yaml"

# Verifica que las imágenes y etiquetas estén organizadas correctamente
assert os.path.exists(DATA_FOLDER), "Data folder does not exist"
assert os.path.exists(LABELS_FOLDER), "Labels folder does not exist"

# Cargar el modelo YOLO
model = YOLO("yolov8n.pt")  # Puedes usar un modelo vacío si lo prefieres

# Iniciar el entrenamiento
results = model.train(
    data=CONFIG_FILE,  # Archivo de configuración con tus clases y rutas de datos
    epochs=100,        # Número de épocas
    imgsz=640          # Tamaño de las imágenes
)

# Guardar el modelo entrenado
model.save("best_model.pt")

print("Training complete. Model saved as 'best_model.pt'.")
