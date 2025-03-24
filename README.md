# Hotel Booking Analytics & Intelligent Q&A System 🏨📊

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Key Features

* **Automated Data Pipeline**
  * CSV ingestion & cleaning
  * Missing value imputation
  * Date normalization
  * Data validation
* **Advanced Analytics**
  * Revenue trend analysis
  * Cancellation pattern detection
  * Geographic distribution mapping
  * Market segment breakdown
* **AI-Powered Q&A**
  * Natural language processing
  * Context-aware responses
  * RAG (Retrieval-Augmented Generation) architecture
  * GPU-accelerated inference
* **Enterprise-Grade API**
  * RESTful endpoints
  * Async request handling
  * JWT authentication (beta)
  * Swagger/OpenAPI documentation

## 🛠 Technologies Used

## 📚 Table of Contents

1. [Installation](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-installation)
2. [Data Preparation](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-data-preparation)
3. [Usage](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-usage)
4. [API Documentation](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-api-documentation)
5. [Deployment](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-deployment)
6. [Testing](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-testing)
7. [Troubleshooting](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-troubleshooting)
8. [Contributing](https://chatgpt.com/c/67dc6f4c-c6dc-8013-8c10-3017622ad359#-contributing)

## 💻 Installation

### Prerequisites

* Python 3.10+
* pip 20.3+
* 4GB RAM minimum
* (Optional) NVIDIA GPU with CUDA 11.x

### Setup

```bash
git clone https://github.com/HiBorn4/LLM_Booking_Analytics_-_QA.git
cd LLM_Booking_Analytics_-_QA

python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt

echo "API_KEY=your_secret_key" > .env
```

## 📂 Data Preparation

```bash
mkdir -p dataset
mv ~/Downloads/hotel_bookings.csv dataset/
python preprocessing.py
```

Output:

* `dataset/hotel_bookings_cleaned.csv`
* `dataset/preprocessing.log`

## 🚀 Usage

```bash
uvicorn app.api:app --reload --port 8000
```

Endpoints:

1. **Swagger UI** : [http://localhost:8000/docs](http://localhost:8000/docs)
2. **Redoc** : [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Sample Queries

```bash
curl http://localhost:8000/analytics?analysis_type=cancellations

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Which month had highest revenue in 2022?"}'
```

## 📡 API Documentation

### POST /ask

**Request:**

```json
{
  "query": "Show hotels with highest cancellation rates",
  "use_gpu": false
}
```

**Response:**

```json
{
  "question": "Show hotels with highest cancellation rates",
  "answer": "Resort hotels show 42% cancellation rate compared to 28% for City hotels...",
  "confidence": 0.87,
  "context": ["Hotel: Resort Hotel, Cancellations: 1423...", ...]
}
```

## 🐳 Deployment

### Docker Setup

```bash
docker build -t booking-analytics .
docker run -p 8000:8000 booking-analytics
```

### Kubernetes (Helm Chart)

```bash
helm install booking-analytics ./charts \
  --set replicaCount=3 \
  --set resources.limits.memory=8Gi
```

## 🧪 Testing

```bash
pytest tests/ --verbose
```

## 🚨 Troubleshooting

| Issue                | Solution                                                 |
| -------------------- | -------------------------------------------------------- |
| CUDA Out of Memory   | Reduce batch size in `rag_hotel_qa`                    |
| CSV Encoding Errors  | Re-save file as UTF-8                                    |
| Missing Dependencies | Re-run `pip install -r requirements.txt`               |
| API Timeouts         | Increase timeout:`uvicorn ... --timeout-keep-alive 30` |

## 🤝 Contributing

```bash
git checkout -b feature/new-analytics
git commit -m 'Add occupancy rate metrics'
git push origin feature/new-analytics
```

Open a pull request.

## 📄 License

MIT License - See [LICENSE](https://chatgpt.com/c/LICENSE) for details.

## 📧 Contact

Maintainer: Hi Born

Email: hiborn4@gmail.com

[Project Board](https://github.com/yourusername/hotel-booking-analytics/projects/1)
