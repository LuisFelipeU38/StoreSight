import pytest
from app import app  # Importa directamente tu aplicación

@pytest.fixture
def client():
    app.config['TESTING'] = True  # Configura el modo de prueba
    with app.test_client() as client:
        yield client

def test_home_page(client):
    # Verifica que la página principal responda correctamente
    response = client.get('/')
    assert response.status_code == 200  # Verifica que el código de estado sea 200 (OK)

def test_invalid_page(client):
    # Verifica que una página inexistente retorne un error 404
    response = client.get('/nonexistent_page')
    assert response.status_code == 404  # Verifica que el código de estado sea 404 (No encontrado)
