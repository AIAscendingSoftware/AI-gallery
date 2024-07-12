'''import requests
import json

def get_admin_ids():
    # Replace with the actual API URL you want to access
    api_url = "http://192.168.29.11:8080/api/facedetection/getAllAdminId"

    # Set headers dictionary (replace with any required headers)
    headers = {
        "Content-Type": "application/json", 
    }

    # Make a GET request to the API endpoint with headers
    response = requests.get(api_url, headers=headers)

    # Check for successful response status code
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()  # Use .json() to directly parse JSON response
    else:
        data = None
    return data

def get_event_ids(admin_id):
    # Replace with the actual API URL you want to access
    api_url = "http://192.168.29.11:8080/api/facedetection/getEventId"

    # Set headers dictionary (replace with any required headers)
    headers = {
        "Content-Type": "application/json",
        "Create-by-id": str(admin_id)  # Replace with admin ID
    }

    # Make a GET request to the API endpoint with headers
    response = requests.get(api_url, headers=headers)

    # Check for successful response status code
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()  # Use .json() to directly parse JSON response
    else:
        data = None

    return data

def get_event_and_user_image(admin_id, event_id):
    # Replace with the actual API URL you want to access
    api_url = "http://192.168.29.11:8080/api/facedetection/getEventFolderImage"

    # Set headers dictionary (replace with any required headers)
    headers = {
        "Content-Type": "application/json",
        "Create-by-id": str(admin_id)
    }
    # Create request body dictionary (replace with your data)
    data = {
        "eventId": event_id,}

    # Make a POST request to the API endpoint with headers and body
    response = requests.get(api_url, headers=headers, json=data)

    # Check for successful response status code
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
    else:
        data = None

    return data

def post_data(data_list):
    api_url = "http://192.168.29.11:8080/api/facedetection/setuserEventImage"
    
    # Set headers dictionary (replace with any required headers)
    headers = {
        "Content-Type": "application/json"
    }
    
    # Send the POST request with data and headers
    response = requests.post(api_url, headers=headers, json=data_list)
    print(response.status_code)
    # Check the response status
    if response.status_code == 200:
        print("Data posted successfully")

    else:
        print("Failed to post data")

'''


import requests

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_admin_ids(self):
        response = requests.get(f"{self.base_url}/getAllAdminId", headers={"Content-Type": "application/json"})
        return response.json() if response.status_code == 200 else None

    def get_event_ids(self, admin_id):
        headers = {"Content-Type": "application/json", "Create-by-id": str(admin_id)}
        response = requests.get(f"{self.base_url}/getEventId", headers=headers)
        return response.json() if response.status_code == 200 else None

    def get_event_and_user_image(self, admin_id, event_id):
        headers = {"Content-Type": "application/json", "Create-by-id": str(admin_id)}
        data = {"eventId": event_id}
        response = requests.get(f"{self.base_url}/getEventFolderImage", headers=headers, json=data)
        return response.json() if response.status_code == 200 else None

    def post_data(self, data_list):
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{self.base_url}/setuserEventImage", headers=headers, json=data_list)
        return response.status_code == 200
