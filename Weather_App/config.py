from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY"   )

Base_URL = "https://api.openweathermap.org/data/2.5/weather"

 