import socket
import json

# Connect to the server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 8000))

# Create the search command
request = {
    "command": "search",
    "query": "machine learning",
    "top_k": 3
}

# Send request
client.send(json.dumps(request).encode('utf-8'))

# Receive and print the response
response = client.recv(4096)
print("Response from server:")
print(json.loads(response))

client.close()
