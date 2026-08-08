"""
Repository Ingestion Pipeline

Indexes target repository code into `code_chunks` using Gemini embeddings.
"""

import ast
import asyncio
import logging
import os
import random
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv

from app.db import db_manager, get_db_pool
from app.models import CodeChunk

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingest")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
VECTOR_DIMENSION = 384


# Extensible Language Extractor Architecture

class BaseLanguageExtractor(ABC):
    """Abstract base class for language-specific code extractors."""

    @abstractmethod
    def extract_chunks(self, file_path: str, code_content: str) -> List[CodeChunk]:
        """Extract functions, async functions, and classes from source code."""
        pass


class PythonLanguageExtractor(BaseLanguageExtractor):
    """AST-based code chunk extractor for Python files."""

    def extract_chunks(self, file_path: str, code_content: str) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        if not code_content or not code_content.strip():
            return chunks

        try:
            tree = ast.parse(code_content)
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
            return chunks
        except Exception as e:
            logger.warning(f"Unexpected error parsing {file_path}: {e}")
            return chunks

        for node in ast.walk(tree):
            chunk_type: Optional[str] = None
            if isinstance(node, ast.AsyncFunctionDef):
                chunk_type = "async_function"
            elif isinstance(node, ast.FunctionDef):
                chunk_type = "function"
            elif isinstance(node, ast.ClassDef):
                chunk_type = "class"

            if chunk_type:
                segment = ast.get_source_segment(code_content, node)
                if not segment:
                    # Line slicing fallback
                    lines = code_content.splitlines()
                    start = getattr(node, "lineno", 1) - 1
                    end = getattr(node, "end_lineno", len(lines))
                    segment = "\n".join(lines[start:end])

                if segment and segment.strip():
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            content=segment.strip(),
                            chunk_type=chunk_type,
                        )
                    )

        return chunks


class ExtractorRegistry:
    """Registry mapping file extensions to language extractors."""

    def __init__(self):
        self._extractors: Dict[str, BaseLanguageExtractor] = {}

    def register(self, extension: str, extractor: BaseLanguageExtractor):
        self._extractors[extension.lower()] = extractor

    def get_extractor(self, file_path: str) -> Optional[BaseLanguageExtractor]:
        ext = os.path.splitext(file_path)[1].lower()
        return self._extractors.get(ext)

    def is_supported(self, file_path: str) -> bool:
        return self.get_extractor(file_path) is not None


# Default Registry Setup
registry = ExtractorRegistry()
registry.register(".py", PythonLanguageExtractor())


# GitHub Repository Fetcher

class GitHubClient:
    """Client for fetching repo tree and raw contents via GitHub API."""

    def __init__(self, repo_full_name: str, token: Optional[str] = None):
        self.repo_full_name = repo_full_name
        self.token = token
        self.base_url = "https://api.github.com"

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_default_branch(self, client: httpx.AsyncClient) -> str:
        url = f"{self.base_url}/repos/{self.repo_full_name}"
        resp = await client.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json().get("default_branch", "main")

    async def get_file_tree(self, client: httpx.AsyncClient, branch: str) -> List[Dict]:
        url = f"{self.base_url}/repos/{self.repo_full_name}/git/trees/{branch}?recursive=1"
        resp = await client.get(url, headers=self._headers())
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        return [item for item in tree if item.get("type") == "blob"]

    async def fetch_file_content(self, client: httpx.AsyncClient, file_path: str, branch: str) -> str:
        url = f"{self.base_url}/repos/{self.repo_full_name}/contents/{file_path}?ref={branch}"
        headers = dict(self._headers())
        headers["Accept"] = "application/vnd.github.v3.raw"
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


# Gemini Embedding Client with Exponential Backoff & Retry Logic

class GeminiEmbeddingClient:
    """Client for generating embeddings using Gemini Embeddings API."""

    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL, dimension: int = VECTOR_DIMENSION):
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={api_key}"

    async def embed_batch(
        self,
        client: httpx.AsyncClient,
        texts: List[str],
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> List[List[float]]:
        """Generate embeddings for a list of text strings in batch with retries."""
        if not texts:
            return []

        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.dimension,
                }
                for t in texts
            ]
        }

        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    self.url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings_data = data.get("embeddings", [])
                    return [item["values"] for item in embeddings_data]
                elif resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        f"Gemini API returned status {resp.status_code}. Retry {attempt + 1}/{max_retries}..."
                    )
                else:
                    resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.warning(f"Network error calling Gemini API: {e}. Retry {attempt + 1}/{max_retries}...")

            if attempt < max_retries - 1:
                delay = (base_delay * (2 ** attempt)) + (random.uniform(0.1, 0.5))
                await asyncio.sleep(delay)

        raise RuntimeError(f"Failed to generate embeddings after {max_retries} attempts.")


# Main Ingestion Orchestrator

class IngestionPipeline:
    """Orchestrates indexing of repository code into code_chunks table."""

    def __init__(
        self,
        repo_full_name: Optional[str] = None,
        github_token: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.repo_full_name = repo_full_name or os.getenv("GITHUB_REPO_FULL_NAME")
        if not self.repo_full_name:
            raise ValueError("GITHUB_REPO_FULL_NAME environment variable is required.")

        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")

        self.github_client = GitHubClient(self.repo_full_name, self.github_token)
        self.embedding_client = GeminiEmbeddingClient(self.gemini_api_key)

    async def _get_or_create_repo_id(self, conn, default_branch: str) -> uuid.UUID:
        """Fetch or insert the repository record in the `repos` table."""
        row = await conn.fetchrow(
            "SELECT id FROM repos WHERE github_full_name = $1", self.repo_full_name
        )
        if row:
            return row["id"]

        repo_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO repos (id, github_full_name, default_branch) VALUES ($1, $2, $3)",
            repo_id,
            self.repo_full_name,
            default_branch,
        )
        logger.info(f"Registered new repo '{self.repo_full_name}' with ID: {repo_id}")
        return repo_id

    async def run(self, batch_size: int = 20):
        """Execute complete ingestion pipeline."""
        logger.info(f"Starting ingestion pipeline for repository: {self.repo_full_name}")
        pool = await get_db_pool()

        async with httpx.AsyncClient() as http_client:
            default_branch = await self.github_client.get_default_branch(http_client)
            logger.info(f"Using default branch: '{default_branch}'")

            async with pool.acquire() as db_conn:
                repo_id = await self._get_or_create_repo_id(db_conn, default_branch)

            file_tree = await self.github_client.get_file_tree(http_client, default_branch)
            supported_files = [
                item["path"] for item in file_tree if registry.is_supported(item["path"])
            ]
            logger.info(f"Found {len(supported_files)} supported source file(s) for indexing.")

            total_chunks_processed = 0

            for file_path in supported_files:
                extractor = registry.get_extractor(file_path)
                if not extractor:
                    continue

                try:
                    raw_content = await self.github_client.fetch_file_content(
                        http_client, file_path, default_branch
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch content for {file_path}: {e}")
                    continue

                chunks = extractor.extract_chunks(file_path, raw_content)
                if not chunks:
                    logger.info(f"No code chunks extracted from {file_path}")
                    continue

                logger.info(f"Extracted {len(chunks)} chunk(s) from {file_path}")

                # Generate embeddings in batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    texts = [c.content for c in batch]
                    embeddings = await self.embedding_client.embed_batch(http_client, texts)

                    for chunk, emb in zip(batch, embeddings):
                        chunk.embedding = emb

                # Save to database (Delete existing file chunks first to avoid duplicates)
                async with pool.acquire() as db_conn:
                    async with db_conn.transaction():
                        # Delete existing chunks for this file
                        await db_conn.execute(
                            "DELETE FROM code_chunks WHERE repo_id = $1 AND file_path = $2",
                            repo_id,
                            file_path,
                        )

                        # Insert newly indexed chunks
                        for chunk in chunks:
                            vec_str = "[" + ",".join(map(str, chunk.embedding)) + "]"
                            await db_conn.execute(
                                """
                                INSERT INTO code_chunks (id, repo_id, file_path, content, embedding, chunk_type)
                                VALUES ($1, $2, $3, $4, $5, $6)
                                """,
                                uuid.uuid4(),
                                repo_id,
                                chunk.file_path,
                                chunk.content,
                                vec_str,
                                chunk.chunk_type,
                            )

                total_chunks_processed += len(chunks)
                logger.info(f"Successfully stored {len(chunks)} chunk(s) for {file_path}")

        logger.info(f"Ingestion complete! Total code chunks indexed: {total_chunks_processed}")


async def main():
    """CLI Entry Point for repository ingestion."""
    pipeline = IngestionPipeline()
    try:
        await pipeline.run()
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())

