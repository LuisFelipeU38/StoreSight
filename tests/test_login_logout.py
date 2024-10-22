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


def test_login_logout(client):
    response = client.post(
        "/login",
        data={"username": "testuser", "password": "password"},
        follow_redirects=True,
    )
    assert b"Logged in successfully!" in response.data

    response = client.get("/logout", follow_redirects=True)
    assert response.status_code == 200


def test_invalid_login(client):
    response = client.post(
        "/login",
        data={"username": "wronguser", "password": "wrongpass"},
        follow_redirects=True,
    )
    assert b"Invalid username or password!" in response.data
