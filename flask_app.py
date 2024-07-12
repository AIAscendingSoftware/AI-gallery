'''from flask import Flask, request, jsonify
from main import wholetrigger,processing_to_analyze_and_post
import threading, requests
app = Flask(__name__)

@app.route('/wholetrigger', methods=['GET'])
def trigger_wholetrigger_function():
    # Send the response immediately
    response = jsonify({"message": "wholetrigger function executed successfully."})

    # Create and start a new thread to run the wholetrigger function
    trigger_thread = threading.Thread(target=wholetrigger)
    trigger_thread.start()  
    return response


def process_response(api_url, headers, data):
    try:
        response = requests.get(api_url, headers=headers, json=data)
        print(response)
        if response.status_code == 200:
            response_data = response.json()
            #put data to 
            processing_to_analyze_and_post(response_data)
        else:
            print(f"Error: GET request failed with status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

@app.route('/singleuserimage/<EventId>/<UserId>', methods=['POST'])
def receive_data(EventId, UserId):
    data = request.get_json() if request.is_json else {}
    data['EventId'] = EventId
    data['UserId'] = UserId
    
    api_url = f"http://192.168.29.11:8080/api/facedetection/getAllEventFolderAndOneUseImage/{UserId}"
    headers = {
        "Content-Type": "application/json",
    }
    data = {"eventId": EventId}
    
    # Start a new thread to process the response
    threading.Thread(target=process_response, args=(api_url, headers, data)).start()
    
    # Return success immediately
    return 'success single user checking with event data has started'

if __name__ == '__main__':
    app.run(host='192.168.29.216', port=8080, debug=True)'''


from flask import Flask, request, jsonify
import threading,requests
from main import FaceRecognitionService

app = Flask(__name__)
service = FaceRecognitionService(base_url="http://136.185.19.60:8083/api/facedetection")

@app.route('/wholetrigger', methods=['GET'])
def trigger_wholetrigger_function():
    response = jsonify({"message": "wholetrigger function executed successfully."})
    trigger_thread = threading.Thread(target=service.whole_trigger)
    trigger_thread.start()
    return response

@app.route('/singleuserimage/<EventId>/<UserId>', methods=['POST'])
def receive_data(EventId, UserId):
    data = request.get_json() if request.is_json else {}
    data['EventId'] = EventId
    data['UserId'] = UserId

    api_url = f"http://192.168.29.11:8080/api/facedetection/getAllEventFolderAndOneUseImage/{UserId}"
    headers = {"Content-Type": "application/json"}
    data = {"eventId": EventId}

    def process_response():
        response = requests.get(api_url, headers=headers, json=data)
        if response.status_code == 200:
            service.processing_to_analyze_and_post(response.json())
        else:
            print(f"Error: GET request failed with status code {response.status_code}")

    threading.Thread(target=process_response).start()
    return 'success single user checking with event data has started'

if __name__ == '__main__':
    app.run(host='192.168.29.216', port=8080, debug=True)

