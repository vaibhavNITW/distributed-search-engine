import socket
import json

# Load query from the JSON file
with open('query.json', 'r') as f:
    query_data = json.load(f)

# Server connection details
HOST = 'localhost'
PORT = 8000  # Make sure your server is running on this port

# Connect to server and send query
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        sock.sendall(json.dumps(query_data).encode('utf-8'))

        # Receive and print response
        response = sock.recv(4096)
        print("Response from server:")
        print(json.loads(response.decode('utf-8')))
except Exception as e:
    print(f"Error: {e}")
