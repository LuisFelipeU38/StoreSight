import pytest
from app import app  # Importa directamente tu aplicación

@pytest.fixture
def client():
    app.config['TESTING'] = True  # Configura el modo de prueba
    with app.test_client() as client:
        yield client

import time

def test_system_performance(client):
    start_time = time.time()
    response = client.get('/analytics')
    end_time = time.time()
    assert (end_time - start_time) < 2.0  # Verifica que la respuesta sea más rápida que 2 segundos

def test_system_performance_high_load(client):
    response = client.get('/analytics', follow_redirects=True)
    assert response.status_code == 200  # Verifica que el código de estado sea 200 (OK)
