import pytest
from app import app 
import io

@pytest.fixture
def client():
    app.config['TESTING'] = True  # Configura el modo de prueba
    with app.test_client() as client:
        yield client

def test_video_upload(client):
    # Simula la carga de un archivo de video válido
    response = client.post('/upload', data={'file': (io.BytesIO(b"video content"), 'test.mp4')})
    assert response.status_code == 302  # Verifica que la respuesta sea una redirección (302)

def test_video_upload_invalid_file_format(client):
    # Simula la carga de un archivo de video no válido
    response = client.post('/upload', data={'file': (io.BytesIO(b"invalid content"), 'test.txt')})
    assert b'Invalid file format. Please upload a video file.' in response.data  # Verifica el mensaje de error
