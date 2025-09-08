# APIgentMan 🕵️‍♂️

**APIgentMan** is an agentic CLI tool for testing and monitoring APIs — like Postman but lighter, autonomous, and terminal-first.
It can run API test collections, monitor performance, detect anomalies, and even generate new tests via an LLM.

---

## ✨ Features

* Run API collections (YAML format) from the CLI
* Assertions on status codes, response time, and payloads
* Performance tracking with simple anomaly detection
* LLM-powered **test generation** (`gen-tests`)
* Monitor mode for repeated runs
* Rich CLI with colored output and reports

---

## 📂 Project Structure

```
.
├── main.py              # APIgentMan CLI entrypoint
├── sample_test_app.py   # Example FastAPI app to test against
├── sample.yaml          # Sample test collection
└── README.md            # You are here
```

---

## ⚡ Setup

### 1. Clone & Install

```bash
git clone https://github.com/dhirajdj30/APIgentMan.git
cd apigentman
uv add fastapi groq httpx pydantic python-dotenv pyyaml rich typer uvicorn
```

### 2. (Optional) LLM Integration

Export your OpenAI key:

```bash
export OPENAI_API_KEY="sk-xxxxx"
```

---

## ▶️ Usage

### 1. Start the Sample API

Run the demo FastAPI server:

```bash
uvicorn sample_test_app:app --reload --port 8000
```

This exposes:

* `GET /ping`
* `GET /items/{item_id}`
* `POST /items`

### 2. Run Tests

Execute the sample test collection:

```bash
python main.py run sample.yaml
```

### 3. Generate Tests with LLM

```bash
python main.py gen-tests --prompt "Generate tests for a weather API with /current and /forecast endpoints"
```

### 4. Monitor Mode

```bash
python main.py monitor sample.yaml --interval 30
```

---

## 🧪 Example `sample.yaml`

```yaml
tests:
  - name: ping test
    method: GET
    url: http://localhost:8000/ping
    expect_status: 200

  - name: get item
    method: GET
    url: http://localhost:8000/items/1
    expect_status: 200
    expect_body_contains: "item"

  - name: create item
    method: POST
    url: http://localhost:8000/items
    json: {"name": "Book"}
    expect_status: 200
    expect_body_contains: "Book"
```

---

## 🔮 Roadmap

* Export performance metrics (Prometheus/Grafana integration)
* Slack/email alerts on anomalies
* Load testing support
* Self-healing test logic
* OpenAPI → test suite auto-generation