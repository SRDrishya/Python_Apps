def main():
    print("Welcome to the Weather App!")
    city = input("Enter the name of the city: ")
    
    if not city:
        print("Please enter a valid city name.")
        return
    print(f"Fetching weather data for {city}...")   


if __name__ == "__main__":
    main()