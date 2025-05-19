import sys
import time
from src.server.server import SearchServer

if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
        server = SearchServer(port=port)

        print("🔍 Welcome to the Distributed Search Engine")
        server.start()

        # Keep the server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()
    except Exception as e:
        print(f"Error: {str(e)}")
