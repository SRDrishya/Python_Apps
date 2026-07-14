import requests
from config import API_KEY, Base_URL

def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(Base_URL, params=params)
    return response