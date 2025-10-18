import pytest
from app import app

@pytest.fixture
def client():
    # Flask provides a test client for making requests without running a server
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_healthcheck(client):
    """Ensure the root endpoint works."""
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'


def test_prediction_endpoint(client):
    """Test the /predict endpoint with valid input."""
    payload = {"temp":0.24,"humidity":0.81,"season_2":0,"season_3":0,"season_4":0,"month_2":0,"month_3":0,"month_4":0,"month_5":0,"month_6":0,"month_7":0,"month_8":0,"month_9":0,"month_10":0,"month_11":0,"month_12":0,"hour_1":0,"hour_2":0,"hour_3":0,"hour_4":0,"hour_5":0,"hour_6":0,"hour_7":0,"hour_8":0,"hour_9":0,"hour_10":0,"hour_11":0,"hour_12":0,"hour_13":0,"hour_14":0,"hour_15":0,"hour_16":0,"hour_17":0,"hour_18":0,"hour_19":0,"hour_20":0,"hour_21":0,"hour_22":0,"hour_23":0,"holiday_1":0,"weekday_1":0,"weekday_2":0,"weekday_3":0,"weekday_4":0,"weekday_5":0,"weekday_6":1,"workingday_1":0,"weather_2":0,"weather_3":0,"weather_4":0}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_missing_json(client):
    """Ensure API returns a 400 if no JSON is provided."""
    response = client.post("/predict")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
