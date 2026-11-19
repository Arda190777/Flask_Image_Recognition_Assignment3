"""Tests for Flask routes and CSRF protection."""

from html.parser import HTMLParser
from io import BytesIO
from unittest.mock import Mock

import app as app_module


class CSRFTokenParser(HTMLParser):
    """Read the CSRF token from the upload form."""

    token = None

    def handle_starttag(self, tag, attrs):
        """Capture the value of the hidden CSRF input."""
        attributes = dict(attrs)
        if tag == "input" and attributes.get("name") == "csrf_token":
            self.token = attributes.get("value")


def get_csrf_token(client):
    """Get a real token using the same client's session."""
    response = client.get("/")
    parser = CSRFTokenParser()
    parser.feed(response.get_data(as_text=True))
    assert parser.token is not None
    return parser.token


def test_home_page_is_accessible(client):
    """The home page should display the upload form."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data
    assert b'name="csrf_token"' in response.data


def test_valid_upload_returns_prediction(client, monkeypatch):
    """An upload with a valid CSRF token should succeed."""
    monkeypatch.setattr(
        app_module, "preprocess_img", Mock(return_value="processed-image")
    )
    monkeypatch.setattr(app_module, "predict_result", Mock(return_value=7))

    response = client.post(
        "/prediction",
        data={
            "csrf_token": get_csrf_token(client),
            "file": (BytesIO(b"valid-image"), "digit.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"7" in response.data
    app_module.preprocess_img.assert_called_once()
    app_module.predict_result.assert_called_once_with("processed-image")


def test_missing_file_returns_error(client):
    """A valid token should not bypass missing-file validation."""
    response = client.post(
        "/prediction",
        data={"csrf_token": get_csrf_token(client)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data


def test_invalid_file_returns_error(client):
    """Invalid image content with a valid token should be rejected."""
    response = client.post(
        "/prediction",
        data={
            "csrf_token": get_csrf_token(client),
            "file": (BytesIO(b"not-an-image"), "invalid.txt"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data


def test_blank_filename_returns_error(client, monkeypatch):
    """An upload without a filename should be rejected before processing."""
    preprocess_mock = Mock(return_value="processed-image")
    predict_mock = Mock(return_value=4)
    monkeypatch.setattr(app_module, "preprocess_img", preprocess_mock)
    monkeypatch.setattr(app_module, "predict_result", predict_mock)

    response = client.post(
        "/prediction",
        data={
            "csrf_token": get_csrf_token(client),
            "file": (BytesIO(b"image-data"), ""),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data
    preprocess_mock.assert_not_called()
    predict_mock.assert_not_called()


def test_missing_csrf_token_is_rejected(client, monkeypatch):
    """A request without a CSRF token must not reach image processing."""
    preprocess_mock = Mock()
    monkeypatch.setattr(app_module, "preprocess_img", preprocess_mock)
    get_csrf_token(client)

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"image-data"), "digit.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"CSRF token is missing" in response.data
    preprocess_mock.assert_not_called()


def test_invalid_csrf_token_is_rejected(client, monkeypatch):
    """A forged CSRF token must not reach image processing."""
    preprocess_mock = Mock()
    monkeypatch.setattr(app_module, "preprocess_img", preprocess_mock)
    get_csrf_token(client)

    response = client.post(
        "/prediction",
        data={
            "csrf_token": "invalid-token",
            "file": (BytesIO(b"image-data"), "digit.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"CSRF token is invalid" in response.data
    preprocess_mock.assert_not_called()