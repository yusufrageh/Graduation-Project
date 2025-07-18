import vonage  # Import the vonage library for sending SMS
import requests  # Import the requests library to make HTTP requests

def get_location_from_my_ip():
    # URL for getting IP location data from ipinfo.io
    ipinfo_url = 'http://ipinfo.io/json'
    
    # Send a GET request to ipinfo.io to get the location data
    response = requests.get(ipinfo_url)
    
    # Parse the JSON response
    data = response.json()

    # Check if 'loc' field is in the response data
    if 'loc' in data:
        # Extract latitude and longitude from the 'loc' field
        loc = data['loc'].split(',')
        latitude = loc[0]
        longitude = loc[1]

        # OpenCage Geocoding API URL to get the address from the coordinates
        opencage_api_key = 'b6b98b8355964988a0afbac84e0b5c23'
        opencage_url = f'https://api.opencagedata.com/geocode/v1/json?q={latitude}+{longitude}&key={opencage_api_key}'

        # Send a GET request to OpenCage API to get geocoding data
        response = requests.get(opencage_url)
        
        # Parse the JSON response from OpenCage
        geodata = response.json()

        # Check if there are any results in the geocoding data
        if geodata and geodata['results']:
            # Get the formatted address from the first result
            address = geodata['results'][0]['formatted']
            
            # Create a Google Maps URL using the latitude and longitude
            google_maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
            
            # Return the Google Maps URL
            return f"Google Maps URL: {google_maps_url}"
        else:
            # Return an error message if no results are found
            return "Location not found"
    else:
        # Return an error message if 'loc' field is not found in the response data
        return "Invalid IP or location data not found"

# Example usage
# Get the location information from the IP address and print it
def sos():
    location_info = get_location_from_my_ip()
    print(location_info)

    # Construct the SOS message
    sos_message = (
        "SOS! The car has stopped and is parked in the rightmost lane. "
        "The driver may be unconscious or asleep. Immediate assistance is required. "
        f"Location details:\n{location_info}"
    )
    # Initialize the Vonage client with your API key and secret
    client = vonage.Client(key="55f1aaf4", secret="6SZb1vjTrM1ED14z")
    sms = vonage.Sms(client)


    # Send an SMS message with the location information
    responseData = sms.send_message(
        {
            "from": "Vonage APIs",  # Sender ID
            "to": "201121041373",  # Recipient phone number
            "text": sos_message,  # Message text
        }
    )

    # Check the status of the sent message and print the result
    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
    return
sos()