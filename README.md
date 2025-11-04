# 🤖 RAG Gemini Microservice - Du học Mentor Pro

Một microservice AI tích hợp **RAG (Retrieval-Augmented Generation)** sử dụng Google Gemini API, cung cấp tư vấn du học thông minh dựa trên dữ liệu tùy chỉnh.

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc](#️-kiến-trúc)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [Chạy ứng dụng](#-chạy-ứng-dụng)
- [API Endpoints](#-api-endpoints)
- [Ví dụ sử dụng](#-ví-dụ-sử-dụng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Troubleshooting](#-troubleshooting)

## ✨ Tính năng

- ✅ **RAG (Retrieval-Augmented Generation)**: Kết hợp dữ liệu tùy chỉnh với Google Gemini
- 🧠 **Quản lý bộ nhớ**: Lưu trữ lịch sử hội thoại riêng cho mỗi người dùng
- 📄 **Upload PDF**: Tải lên và tự động indexing tài liệu PDF
- 🔐 **Xác thực**: Yêu cầu `X-Internal-Key` header cho các API requests
- 🌐 **Vector Store**: Sử dụng FAISS hoặc numpy cho similarity search
- 🔄 **Cân bằng model**: Hỗ trợ nhiều model Gemini và nhiều user
- 📍 **Embedding linh hoạt**: Hỗ trợ sentence-transformers hoặc Google Generative AI
- 🚀 **FastAPI**: Hiệu suất cao, async/await support

## 🏗️ Kiến trúc

```
┌─────────────────────┐
│   FastAPI Server    │
├─────────────────────┤
│  /query             │ → Xử lý query + RAG
│  /upload-doc        │ → Tải lên tài liệu
│  /healthz           │ → Health check
└─────────────────────┘
         │
    ┌────┴────┬─────────────────┐
    │          │                 │
┌───▼──┐  ┌──▼────┐      ┌──────▼─────┐
│ Bot  │  │Memory │      │ Retriever  │
│Cache │  │Manager│      │(RAG)       │
└───┬──┘  └──┬────┘      └──────┬─────┘
    │       │                    │
    └───┬───┴─────────┬──────────┘
        │             │
    ┌───▼──┐      ┌──▼──────┐
    │ JSON │      │ Vector  │
    │Files │      │ Store   │
    └──────┘      └─────────┘
```

## 🔧 Yêu cầu hệ thống

- **Python**: 3.9 hoặc cao hơn
- **pip**: Package manager cho Python
- **Git**: Để clone repository (tùy chọn)

## 📦 Cài đặt

### 1. Clone hoặc tải project

```bash
cd /Users/builong/Develop/microservice
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt (macOS/Linux)
source venv/bin/activate

# Kích hoạt (Windows)
# venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Ghi chú**: Nếu gặp lỗi khi cài `google-genai`, hãy kiểm tra phiên bản Python của bạn (≥3.9):
```bash
python3 --version
```

### 4. (Tùy chọn) Cài thêm công cụ hỗ trợ

Để có feedback tiến độ khi indexing tài liệu:
```bash
pip install tqdm
```

## ⚙️ Cấu hình

### 1. Tạo file `.env`

Tại thư mục gốc project, tạo file `.env`:

```bash
# Google Generative AI
GEMINI_API_KEY=your_google_api_key_here
# Hoặc sử dụng alias
GOOGLE_API_KEY=your_google_api_key_here

# Microservice
MICROSERVICE_INTERNAL_KEY=your_secret_key_here
ASSISTANT_NAME=Du học Mentor Pro
GEMINI_DEFAULT_MODEL=gemini-2.5-flash-lite
GEMINI_DEFAULT_TEMPERATURE=0.2

# Thư mục
DOCS_DIRECTORY=data/docs
MEMORY_DIRECTORY=data/memory
VECTOR_STORE_PATH=data/vector_store.pkl

# Chat history
CHAT_HISTORY_TURNS=10

# (Tùy chọn) Embedding model
SENTENCE_TRANSFORMERS_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 2. Lấy Google API Key

1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikeys)
2. Click "Create API Key"
3. Copy key vào file `.env`

### 3. Đặt Microservice Key

Tạo một secret key mạnh (ví dụ: 32 ký tự random):
```bash
openssl rand -base64 24
```

Paste vào `MICROSERVICE_INTERNAL_KEY` trong `.env`.

## 🚀 Chạy ứng dụng

### Chế độ development

```bash
# Chạy với auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Hoặc sử dụng python
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Server sẽ khởi động tại `http://localhost:8000`

### Xem API documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Chế độ production

```bash
# Chạy với Gunicorn (nếu cần)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

### Chạy bằng Docker

```bash
docker build -t rag-gemini .
docker run --rm -it \
  --env-file .env \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  rag-gemini
```

> � Nhớ tạo sẵn file `.env` và thư mục `data/` trong máy host để container đọc được cấu hình, bộ nhớ hội thoại và tài liệu.

## �📡 API Endpoints

### 1. Health Check

```http
GET /healthz
Header: X-Internal-Key: <your_key>
```

**Response:**
```json
{
  "status": "ok"
}
```

### 2. Upload Document (PDF)

```http
POST /upload-doc
Header: X-Internal-Key: <your_key>
Content-Type: multipart/form-data

file: <binary_pdf_file>
filename: (optional) custom_name.pdf
```

**Response:**
```json
{
  "stored_path": "data/docs/document_name.pdf",
  "indexed_chunks": 15
}
```

### 3. Query (RAG)

```http
POST /query
Header: X-Internal-Key: <your_key>
Content-Type: application/json

{
  "query": "Chi phí học đại học ở Canada bao nhiêu?",
  "model": "gemini-2.5-flash-lite",
  "temperature": 0.2,
  "top_k": 3,
  "user_id": "user_001",
  "prompt_template": null
}
```

**Response:**
```json
{
  "answer": "Theo dữ liệu hiện có...",
  "sources": []
}
```

## 💡 Ví dụ sử dụng

### Ví dụ 1: Health Check

```bash
curl -X GET http://localhost:8000/healthz \
  -H "X-Internal-Key: your_secret_key"
```

### Ví dụ 2: Upload tài liệu

```bash
curl -X POST http://localhost:8000/upload-doc \
  -H "X-Internal-Key: your_secret_key" \
  -F "file=@/path/to/document.pdf" \
  -F "filename=study_guide.pdf"
```

### Ví dụ 3: Query với RAG

```bash
curl -X POST http://localhost:8000/query \
  -H "X-Internal-Key: your_secret_key" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: student_123" \
  -d '{
    "query": "Du học Mỹ cần điều kiện gì?",
    "model": "gemini-2.5-flash-lite",
    "temperature": 0.2,
    "top_k": 3
  }'
```

### Ví dụ 4: Python Client

```python
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "your_secret_key"
USER_ID = "student_123"

headers = {
    "X-Internal-Key": API_KEY,
    "X-User-Id": USER_ID,
}

# Query
response = requests.post(
    f"{BASE_URL}/query",
    headers=headers,
    json={
        "query": "Du học Úc cần chuẩn bị gì?",
        "top_k": 3,
        "temperature": 0.2,
    }
)

print(response.json())
```

## 📁 Cấu trúc thư mục

```
microservice/
├── app.py                      # FastAPI application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (ignore)
├── README.md                   # Tài liệu này
├── .gitignore                  # Git ignore rules
│
├── chatbot/
│   ├── __init__.py
│   ├── gemini_bot.py          # GeminiBot class (AI logic)
│   ├── memory.py              # MemoryManager (conversation history)
│   ├── retriever.py           # RAGRetriever (vector search)
│   └── utils.py               # Utility functions
│
└── data/
    ├── docs/                  # Uploaded PDF documents
    ├── memory/                # User conversation histories (JSON)
    └── vector_store.pkl       # FAISS index + embeddings
```

### Chi tiết các component

| File | Mô tả |
|------|-------|
| `app.py` | FastAPI server, route handlers, request validation |
| `gemini_bot.py` | GeminiBot class - nối dây RAG với Gemini API, quản lý prompt |
| `memory.py` | MemoryManager - lưu trữ hội thoại multi-user |
| `retriever.py` | RAGRetriever - embedding, vector indexing, similarity search |

## 🔍 Troubleshooting

### 1. Lỗi: "GEMINI_API_KEY not set"

**Nguyên nhân**: Google API key chưa được cấu hình

**Giải pháp**:
```bash
# Kiểm tra .env
cat .env | grep GEMINI_API_KEY

# Nếu chưa có, thêm vào
echo "GEMINI_API_KEY=your_key_here" >> .env
```

### 2. Lỗi: "Missing X-Internal-Key header"

**Nguyên nhân**: Request không có header xác thực

**Giải pháp**: Thêm header vào request
```bash
curl -H "X-Internal-Key: your_secret_key" ...
```

### 3. Lỗi: "No matching distribution found for google-genai"

**Nguyên nhân**: Python version < 3.9

**Giải pháp**: Cập nhật Python
```bash
python3 --version  # Kiểm tra phiên bản
# Cài Python 3.10+
```

### 4. Lỗi: "Failed to initialize GenerativeModel"

**Nguyên nhân**: Model name không hợp lệ

**Giải pháp**: Kiểm tra model được hỗ trợ:
- `gemini-2.5-flash-lite` (mặc định)
- `gemini-2.5-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### 5. Lỗi: "FAISS import failed"

**Nguyên nhân**: FAISS chưa cài đặt (tùy chọn)

**Giải pháp**: Hệ thống sẽ tự động fallback sang numpy, hoặc cài FAISS:
```bash
pip install faiss-cpu
```

### 6. Lỗi: "sentence-transformers not available"

**Nguyên nhân**: Embedding backend chưa cài

**Giải pháp**: Cài sentence-transformers hoặc Google genai API sẽ tự động dùng
```bash
pip install sentence-transformers
```

### 7. Lỗi: PDF extraction không hoạt động

**Nguyên nhân**: pypdf chưa cài

**Giải pháp**:
```bash
pip install pypdf
```

## 🔐 Bảo mật

### Best Practices

1. **Không commit `.env` vào Git**
   ```bash
   # Đã có trong .gitignore
   .env
   .env.*
   ```

2. **Sử dụng Secret Manager** cho production:
   - AWS Secrets Manager
   - Google Cloud Secret Manager
   - HashiCorp Vault

3. **Rotate API keys** định kỳ

4. **Giới hạn rate limiting** (có thể thêm vào future)

## 📊 Monitoring & Logging

Ứng dụng có sẵn logging tích hợp:

```python
# Xem logs
tail -f app.log  # (nếu chạy production)

# Hoặc từ stdout khi chạy development
# Logs sẽ hiển thị trực tiếp
```

**Log levels**:
- `INFO`: Request/response, model initialization
- `WARNING`: API key missing, fallback to alternative
- `ERROR`: Failed to process, missing dependencies
- `DEBUG`: Detailed operation traces

## 🚧 Development

### Thêm tính năng mới

1. Sửa đổi route trong `app.py`
2. Update `chatbot/` modules nếu cần
3. Test với `/docs` endpoint
4. Cập nhật README

### Chạy tests (nếu có)

```bash
pytest tests/ -v
```

## 📝 Phiên bản

- **Version**: 1.0.0
- **Last Updated**: 2025-10-21
- **Python**: 3.9+
- **FastAPI**: 0.115.0+

## 🤝 Support

Nếu gặp vấn đề:

1. Kiểm tra logs
2. Xem phần [Troubleshooting](#-troubleshooting)
3. Kiểm tra `.env` configuration
4. Đảm bảo tất cả dependencies đã cài

## 📚 Tài liệu tham khảo

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [Langchain Text Splitters](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**Được tạo bởi**: Du học Mentor Pro Team  
**License**: Proprietary

## ☁️ Triển khai GitHub Actions & Amazon EC2

- **Workflow CI/CD**: `./.github/workflows/deploy-ec2.yml` build Docker image, push lên Amazon ECR và SSH lên EC2 để chạy container mới.
- **Secrets bắt buộc** (trong GitHub repo → Settings → Secrets and variables → Actions):
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
  - `ECR_REPOSITORY` (ví dụ `rag-gemini-service`)
  - `EC2_HOST` (public IP hoặc domain), `EC2_USER` (thường là `ec2-user` hoặc `ubuntu`), `EC2_SSH_KEY` (private key dạng PEM), tùy chọn `EC2_SSH_PORT` nếu khác 22.
- **Chuẩn bị trên EC2**:
  - Cài Docker và AWS CLI (`sudo yum install -y docker awscli` hoặc tương đương, bật `docker` service và thêm user vào group nếu cần).
  - Tạo thư mục `/opt/microservice`, copy file `.env` chứa API key, microservice key, cấu hình khác.
  - Mở port 8000 (hoặc port bạn map) trong Security Group.
- **Chu kỳ deploy**: mỗi lần `git push` lên branch `main` hoặc chạy `workflow_dispatch`, GitHub Actions sẽ build image, cập nhật tag mới nhất trên ECR và khởi động lại container `rag-gemini` trên EC2.
- **Tuỳ biến**: sửa script SSH trong workflow để đổi port, mount volume khác, hoặc thêm lệnh migrate trước khi `docker run`.
