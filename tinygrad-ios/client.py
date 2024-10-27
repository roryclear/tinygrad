import requests

# Replace 'your-iphone-ip' with your iPhone's IP address
url = "http://192.168.1.105:8081"

# JSON payload to send
payload = {"queue":[{"data": "Hello from Python client!"},{"data2": "Hello from Python client!2"}]}

# Send JSON data with POST request
response = requests.post(url, json=payload)

# Check response
if response.status_code == 200:
    print("response =",response.text)