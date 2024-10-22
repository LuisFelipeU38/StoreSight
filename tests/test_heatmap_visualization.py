import pytest
from app import app, db, User
from werkzeug.security import generate_password_hash


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            user = User(
                username="testuser", password_hash=generate_password_hash("password")
            )
            db.session.add(user)
            db.session.commit()
            yield client
            db.session.remove()
            db.drop_all()


def login(client):
    return client.post(
        "/login",
        data={"username": "testuser", "password": "password"},
        follow_redirects=True,
    )


def test_heatmap_display(client):
    login(client)
    with client.session_transaction() as sess:
        sess["scatter_plot"] = "scatter_plot.png"
    response = client.get("/analytics", follow_redirects=True)
    print(response.data)  # Debugging line
    assert b"Scatter Plot" in response.data


def test_heatmap_error(client):
    login(client)
    response = client.get("/analytics", follow_redirects=True)
    assert response.status_code == 200
    assert b"Scatter Plot" not in response.data
