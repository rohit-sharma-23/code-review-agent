# AI Code Review Agent (Backend)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.0/Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![Supabase pgvector](https://img.shields.io/badge/Supabase-pgvector(384d)-3ECF8E?style=flat&logo=supabase&logoColor=white)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, automated AI Code Review System powered by **FastAPI**, **Google Gemini LLM**, **Model Context Protocol (MCP)**, and **Supabase pgvector RAG**. The system automatically triggers when a GitHub Pull Request is opened or updated, fetches codebase context using vector similarity search, performs structured code analysis, and posts inline review comments directly onto GitHub PR lines.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture Diagram](#architecture-diagram)
- [Tech Stack](#tech-stack)
- [How the PR Flow Works](#how-the-pr--ai-review--github-comment-flow-works)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Local Setup Instructions](#local-setup-instructions)
- [Environment Variables](#environment-variables)
- [Testing Instructions](#testing-instructions)
- [Deployment Instructions](#deployment-instructions)
- [API Reference](#api-reference)
- [Future Work](#future-work)

---

## Project Overview

Reviewing Pull Requests manually can be slow and inconsistent. Traditional static analysis tools lack full repository context and cannot provide human-like architectural suggestions.

This project solves that by combining **Retrieval-Augmented Generation (RAG)** with the **Model Context Protocol (MCP)**:
- **AST-Based Indexing:** Parses target repository source files into functions and classes using Python's `ast` module, generating 384-dimensional Gemini embeddings stored in PostgreSQL (`pgvector`).
- **Context-Aware Reviews:** When a PR is opened, the agent extracts modified diff hunks, queries pgvector for relevant repository context, and feeds both the diff and RAG context to Gemini LLM.
- **Automated Inline Feedback:** Posts precise review comments (categorized as `bug_risk`, `missing_test`, or `style_deviation`) onto added PR lines with line position validation and auto-snapping tolerance.

---

## Architecture Diagram

![Architecture Diagram](./static/diagram.svg)



---

## Tech Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Framework** | FastAPI (Python 3.10+) | Asynchronous high-performance Web API & Webhook engine |
| **AI / LLM** | Google Gemini (`gemini-flash-latest`) | Fast, structured code review generation |
| **Embeddings** | Gemini (`gemini-embedding-001`) | 384-dimensional vector embedding generation |
| **Vector DB** | Supabase PostgreSQL (`pgvector`) | Store and query semantic code chunk embeddings |
| **Database Pool** | `asyncpg` | Async PostgreSQL connection pool management |
| **MCP Integration** | Python MCP SDK (`MCPServer`) | Standardized tool exposure for codebase retrieval & GitHub API |
| **Validation** | Pydantic v2 | Data schema validation for review findings and feedback |
| **Testing** | FastAPI `TestClient`, `httpx` | Integration testing for webhooks, agent loop, and APIs |

---

## How the PR → AI Review → GitHub Comment Flow Works

1. **Webhook Event Received (`POST /webhook`):**
   - GitHub sends a webhook payload when a Pull Request action occurs (`opened`, `synchronize`, `reopened`).
   - The endpoint verifies the `X-Hub-Signature-256` header against `GITHUB_WEBHOOK_SECRET` using HMAC-SHA256 constant-time comparison.
   - The handler immediately responds with `HTTP 200 Accepted` (~50ms) and schedules `review_pr()` as a FastAPI background task.

2. **Idempotency & Diff Parsing:**
   - `review_pr()` attempts to insert a record into the `reviews` table.
   - A PostgreSQL constraint `UNIQUE (repo_id, pr_number, commit_sha)` prevents duplicate reviews if the same delivery or commit SHA is received multiple times.
   - The agent calls `get_diff()` to fetch the unified PR diff and parses it into per-file change hunks.

3. **Concurrent RAG Retrieval:**
   - The agent concurrently queries `search_codebase()` for each modified file hunk using `asyncio.gather()` bounded by `asyncio.Semaphore(5)`.
   - `search_codebase()` generates a 384-d embedding of the diff hunk and executes a pgvector cosine similarity search (`ORDER BY embedding <=> $1::vector LIMIT 3`).

4. **Structured Gemini LLM Prompting:**
   - Constructs a prompt containing PR metadata, diff hunks, retrieved codebase context, review instructions, and strict JSON output requirements.
   - Prompts Gemini (`gemini-flash-latest`) and validates output objects using Pydantic (`ReviewFinding`).

5. **Inline Comment Validation & Posting:**
   - For each finding, `post_review_comment()` parses added lines from the diff (`+`).
   - Includes a **±2 line auto-snapping tolerance algorithm** to adjust minor LLM line miscalculations onto valid added lines.
   - Posts the comment to GitHub via the PR Review Comments API and persists the record into the `comments` table.

---

## Key Engineering Decisions

### 1. AST-Based Code Chunking (`app/ingest.py`)
Rather than relying on primitive line-number or chunk-size splitters that break function bodies in half, we implemented Python's native `ast` parser. Chunks are extracted at structural boundaries (`FunctionDef`, `AsyncFunctionDef`, `ClassDef`) using `ast.get_source_segment()`, preserving complete code semantics for embeddings.

### 2. Supabase pgvector with 384-Dimensional Embeddings
To maintain low latency and fit PostgreSQL vector column constraints (`vector(384)`), embedding requests to Gemini (`gemini-embedding-001`) explicitly specify `"outputDimensionality": 384`. Cosine similarity queries are executed natively in PostgreSQL using the `<=>` vector distance operator.

### 3. Model Context Protocol (MCP) Server Architecture (`app/mcp_server.py`)
GitHub operations and RAG search tools are encapsulated within an MCP Server implementation. This decouples retrieval and GitHub API logic from the agent execution loop, providing clean, reusable tools (`get_diff`, `get_file_context`, `search_codebase`, `post_review_comment`).

### 4. Non-Blocking Fast Webhook Engine
AI LLM reviews take 10–15 seconds to execute. To prevent GitHub webhook delivery timeouts (which occur after 10s), `POST /webhook` verifies the signature, resolves the repository ID, dispatches `review_pr()` into FastAPI's `BackgroundTasks`, and returns `200 Accepted` immediately.

---

## Local Setup Instructions

### Prerequisites
- Python 3.10 or higher
- Git
- PostgreSQL database with `pgvector` extension enabled (or Supabase PostgreSQL instance)
- Google Gemini API Key
- GitHub Personal Access Token (PAT) with repository read/write permissions

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rohitsrma/code-review-agent.git
   cd code-review-agent/backend
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file inside the `backend` directory (see [Environment Variables](#environment-variables)).

5. **Run Database Ingestion (Index Target Repository):**
   ```bash
   python -m app.ingest
   ```

6. **Start the FastAPI Development Server:**
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

7. **Access API Documentation:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to view the interactive OpenAPI documentation.

---

## Environment Variables

Create a `backend/.env` file:

```env
# GitHub Configuration
GITHUB_TOKEN=ghp_your_github_personal_access_token
GITHUB_REPO_FULL_NAME=rohitsrma/Resume-reviewer
GITHUB_WEBHOOK_SECRET=your_secure_webhook_secret_key

# Supabase PostgreSQL Vector Database
DATABASE_URL=postgresql://postgres:your_password@db.your_project.supabase.co:5432/postgres

# Google Gemini AI Configuration
GEMINI_API_KEY=AIzaSy_your_gemini_api_key
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
EMBEDDING_MODEL=gemini-embedding-001

# CORS Configuration (Optional for frontend integration)
FRONTEND_URL=http://localhost:3000
```

---

## Testing Instructions

The codebase includes integration test suites:

### 1. Run Webhook Receiver Tests
Tests HMAC signature verification (missing, invalid, valid), event filtering (`pull_request`, `ping`), PR action filtering (`opened`, `synchronize`, `reopened`), and payload format parsing (`application/json` and `application/x-www-form-urlencoded`):
```bash
python test_webhook.py
```

### 2. Run End-to-End Agent Loop Test
Executes a full code review workflow against a real PR diff, testing RAG context retrieval, Gemini LLM generation, Pydantic validation, comment posting, and review idempotency:
```bash
python test_agent_workflow.py
```

### 3. Run Dashboard API Tests
Tests REST endpoints (`GET /api/reviews`, `GET /api/reviews/{id}`, `POST /api/comments/{id}/feedback`, `GET /api/metrics`):
```bash
python test_dashboard_api.py
```

---

## Deployment Instructions

### Deploying to Render

1. Create a new **Web Service** on Render connected to your GitHub repository.
2. Set **Root Directory** to `backend`.
3. Configure settings:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add all environment variables listed in `.env` under Render Environment settings.
5. Deploy and note your public service URL (e.g. `https://code-review-agent.onrender.com`).

### GitHub Webhook Registration

1. Go to your target GitHub repository → **Settings** → **Webhooks** → **Add webhook**.
2. **Payload URL:** `https://code-review-agent.onrender.com/webhook`
3. **Content type:** `application/json` (or `application/x-www-form-urlencoded`)
4. **Secret:** Set to your `GITHUB_WEBHOOK_SECRET`
5. **Events:** Select **Pull requests** (`opened`, `synchronize`, `reopened`).

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status & PostgreSQL pool connection test |
| `POST` | `/webhook` | GitHub Pull Request webhook receiver |
| `GET` | `/api/reviews` | Paginated review history (`?page=1&page_size=20`) |
| `GET` | `/api/reviews/{id}` | Detailed review info with generated comments |
| `POST` | `/api/comments/{id}/feedback` | Submit developer helpfulness rating (`{"was_helpful": true}`) |
| `GET` | `/api/metrics` | System-wide analytics & daily helpfulness trend history |

---

## Future Work

- **Multi-Language AST Extractors:** Expand language extractor strategies (`ExtractorRegistry`) to parse JavaScript/TypeScript (`@babel/parser` or Tree-sitter), Go, and Java source files.
- **PR Auto-Summarization:** Add high-level PR summaries posted directly as top-level PR issue comments in addition to inline code comments.
- **Fine-Tuned Prompt Templates:** Support per-repository review guidelines (e.g., enforcing company-specific linter standards or security rules via custom config files).
