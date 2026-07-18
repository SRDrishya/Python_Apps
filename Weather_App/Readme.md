# Weather App

A Python command-line weather application that fetches current weather details for a city using the OpenWeather API.

## Features
- Accepts a city name from the user.
- Retrieves weather information from the OpenWeather API.
- Displays temperature, humidity, pressure, weather description, wind speed, and visibility.

## How to Run
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file and add your API key:
   ```env
   OPENWEATHER_API_KEY=your_api_key_here
   ```
3. Run the app:
   ```bash
   python main.py
   ```
4. Enter the name of the city when prompted.

## Files
- `main.py` – entry point for the application.
- `weather_service.py` – handles API requests.
- `display.py` – formats and prints the weather information.
- `config.py` – loads the configuration and API key.

## Requirements
- Python 3.x
- `requests`
- `python-dotenv`
