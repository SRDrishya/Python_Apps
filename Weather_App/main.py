import requests
from weather_service import get_weather

from display import display_weather
def main():
    city = input("Enter city name: ")
    response = get_weather(city)
    if response.status_code == 200:
        weather_data = response.json()
        display_weather(weather_data)
    else:
        print(f"Error fetching weather data: {response.json()['message']}")

if __name__ == "__main__":
    main()