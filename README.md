# 🔎 Distributed Search Engine

 
A beginner-friendly Python-based **Distributed Search Engine** that demonstrates the foundations of building scalable, modular search systems using **sockets**, **ranking algorithms**, and **client-server architecture**. This project simulates how search engines work internally while making it easy for beginners to understand.

---

## 📌 Features

* 🔌 TCP/IP-based Client-Server communication
* 🔍 Query handling and ranking of dummy documents
* 🧮 Simple scoring mechanism to simulate search ranking
* 🗂️ Modular structure (server, searcher, ranker)
* 🧪 JSON-based testing client

---

## 🧰 Tools & Technologies

| Category        | Tools/Tech         |
| --------------- | ------------------ |
| Language        | Python 3.x         |
| Communication   | Socket Programming |
| System          | Unix/Linux         |
| Retrieval Logic | Custom Ranking     |
| Version Control | Git, GitHub        |

---

## 📂 Project Structure

```
search-engine/
├── src/
│   ├── server/
│   │   └── server.py
│   ├── ranker/
│   │   ├── ranker.py
│   ├── searcher/
│   │   └── searcher.py
│   └── main.py
├── client_test.py
├── json_client.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Setup

> No prior experience required. Follow these simple steps:

### 1. Clone the Repository

```bash
git clone https://github.com/vaibhavNITW/distributed-search-engine.git
cd distributed-search-engine
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Python Path

```bash
export PYTHONPATH=$(pwd)
```

---

## 🚦 How to Run the Project

### Start the Server:

```bash
python3 src/main.py 8000
```

Expected Output:

```
🔥 Ranker script is starting...
🔍 Welcome to the Distributed Search Engine
Server started on 0.0.0.0:8000
```

### Send a Query:

Open a new terminal and run:

```bash
python3 json_client.py
```

You should see something like:

```json
{
  "status": "ok",
  "query": "machine learning",
  "results": [
    {"document": "Dummy doc 1"},
    {"document": "Dummy doc 2"},
    {"document": "Dummy doc 3"},
    ...
  ],
  "count": 5,
  "timestamp": "..."
}
```

---

## 🖼️ Screenshots

### ✅ Server Running

![Server Running](screenshots/serverstarted.png)

### ❌ Server Closed

![Server Closed](screenshots/serverclosed.png)

### 🔍 Client Query Result

![Client Response](screenshots/client.png)

## 🧠 Purpose of the Project

To help absolute beginners understand:

* 🔌 How distributed systems exchange data over sockets
* 🧮 How ranking logic can be implemented
* 🔧 How modular architecture supports scalability
* 🧪 How client-server models are built and tested

This is **not a full-fledged web search engine**, but it closely mimics the core design and flow of one.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to your branch
5. Create a pull request

---

## 📜 License

This project is licensed under the MIT License.
