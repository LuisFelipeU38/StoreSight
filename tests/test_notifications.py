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


def test_upload_success(client):
    login(client)
    with client.session_transaction() as sess:
        sess["scatter_plot"] = "scatter_plot.png"

    response = client.get("/analytics", follow_redirects=True)
    print(response.data)  # Debugging line
    assert response.status_code == 200
    assert b"Scatter Plot" in response.data


def test_upload_failure(client):
    login(client)
    response = client.get("/analytics", follow_redirects=False)
    print(response.data)  # Debugging line
    assert response.status_code == 302  # Redirige al home si no hay scatter plot
    assert b"Redirecting..." in response.data
