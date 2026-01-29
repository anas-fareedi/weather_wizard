from core.weather_wizard import WeatherWizard

def main():
    try:
        wizard = WeatherWizard()
        wizard.run()
    except Exception as e:
        print(f"Failed to start: {str(e)}")

if __name__ == "__main__":
    main()