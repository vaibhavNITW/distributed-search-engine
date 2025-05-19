import socket
import json
import threading
import logging
import sys
import os
from datetime import datetime

# Add the project root to the Python path to enable relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import search engine components
from src.searcher.searcher import Searcher
from src.ranker.ranker import MLRanker

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SearchServer:
    def __init__(self, host='0.0.0.0', port=9000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.searcher = Searcher()
        self.ranker = MLRanker()
        self.running = False

        # Load ML ranker model if available
        self.ranker.load_model()

    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            logging.info(f"Server started on {self.host}:{self.port}")
            print(f"Server started on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    logging.info(f"Client connected: {client_address}")
                    print(f"Client connected: {client_address}")

                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                    self.clients.append((client_socket, client_address, client_thread))

                except Exception as e:
                    if self.running:
                        logging.error(f"Error accepting client connection: {str(e)}")
                        print(f"Error accepting client: {str(e)}")  # <--- NEW PRINT

        except Exception as e:
            logging.error(f"Server error: {str(e)}")
            print(f"Server error: {str(e)}")  # <--- NEW PRINT
        finally:
            self.stop()

    def stop(self):
        self.running = False

        for client_socket, _, _ in self.clients:
            try:
                client_socket.close()
            except:
                pass

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        logging.info("Server stopped")
        print("Server stopped")

    def handle_client(self, client_socket, client_address):
        try:
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    break

                request = json.loads(data.decode('utf-8'))
                response = self.process_request(request)
                client_socket.send(json.dumps(response).encode('utf-8'))

                query = request.get('query', '')
                if query:
                    log_entry = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'client': client_address[0],
                        'query': query,
                        'num_results': len(response.get('results', []))
                    }
                    self.log_search(log_entry)

        except Exception as e:
            logging.error(f"Error handling client {client_address}: {str(e)}")
            print(f"Error handling client {client_address}: {str(e)}")  # <--- NEW PRINT
        finally:
            try:
                client_socket.close()
                logging.info(f"Client disconnected: {client_address}")
                print(f"Client disconnected: {client_address}")
            except:
                pass
            self.clients = [c for c in self.clients if c[0] != client_socket]

    def process_request(self, request):
        command = request.get('command', '')
        if command == 'search':
            return self.search_command(request)
        elif command == 'ping':
            return {'status': 'ok', 'message': 'pong'}
        else:
            return {'status': 'error', 'message': f"Unknown command: {command}"}

    def search_command(self, request):
        query = request.get('query', '')
        top_k = request.get('top_k', 10)

        if not query:
            return {'status': 'error', 'message': 'Query parameter is required'}

        try:
            results = self.searcher.search(query, top_k)
            if self.ranker.model and results:
                ml_ranked = self.ranker.rank(query, results)
                results = [item['document'] for item in ml_ranked]

            return {
                'status': 'ok',
                'query': query,
                'results': results,
                'count': len(results),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logging.error(f"Error processing search query '{query}': {str(e)}")
            print(f"Error processing query '{query}': {str(e)}")  # <--- NEW PRINT
            return {'status': 'error', 'message': str(e)}

    def log_search(self, log_entry):
        log_file = 'logs/search_logs.jsonl'
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logging.error(f"Error logging search: {str(e)}")
            print(f"Error logging search: {str(e)}")  # <--- NEW PRINT


# Entry point
if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
        server = SearchServer(port=port)
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        logging.info("Server shutdown requested by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Error starting server: {str(e)}")
