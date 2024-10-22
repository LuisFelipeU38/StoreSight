import pytest
from app import app, db, User
from werkzeug.security import generate_password_hash
import io


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


def test_valid_video_upload(client):
    login(client)
    # Open the test video file
    with open("tests/testvideo.mp4", "rb") as video_file:
        response = client.post(
            "/upload",
            data={"file": (video_file, "testvideo.mp4")},
            follow_redirects=True,
        )
    assert b"The video has been uploaded and processed successfully" in response.data


def test_invalid_video_upload(client):
    login(client)
    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"content"), "invalid.txt")},
        follow_redirects=True,
    )
    assert b"Invalid file format" in response.data
