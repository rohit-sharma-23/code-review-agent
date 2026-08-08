"""
MCP Server implementation exposing GitHub operations and codebase retrieval tools.
"""

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from app.db import db_manager, get_db_pool

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mcp_server")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
VECTOR_DIMENSION = 384

mcp_server = MCPServer("code-review-mcp-server")

# Helper Utilities & GitHub API Client

class GitHubService:
    """Service encapsulating GitHub REST API operations."""

    def __init__(self, repo_full_name: Optional[str] = None, token: Optional[str] = None):
        self.repo_full_name = repo_full_name or os.getenv("GITHUB_REPO_FULL_NAME", "")
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.base_url = "https://api.github.com"

    def _headers(self, accept: str = "application/vnd.github+json") -> Dict[str, str]:
        headers = {"Accept": accept}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_pr_details_and_diff(self, pr_number: int) -> Dict[str, Any]:
        """Fetch PR metadata and unified diff from GitHub."""
        if not self.repo_full_name:
            return {
                "status": "error",
                "error": "GITHUB_REPO_FULL_NAME environment variable is not configured.",
                "code": "CONFIG_ERROR"
            }

        async with httpx.AsyncClient() as client:
            pr_url = f"{self.base_url}/repos/{self.repo_full_name}/pulls/{pr_number}"
            
            try:
                pr_resp = await client.get(pr_url, headers=self._headers())
            except httpx.RequestError as e:
                return {
                    "status": "error",
                    "error": f"Network error connecting to GitHub: {str(e)}",
                    "code": "NETWORK_ERROR"
                }

            if pr_resp.status_code == 404:
                return {
                    "status": "error",
                    "error": f"Pull request #{pr_number} not found in repository '{self.repo_full_name}'.",
                    "code": "PR_NOT_FOUND"
                }
            elif pr_resp.status_code in (401, 403):
                return {
                    "status": "error",
                    "error": f"GitHub authentication or permission failure (HTTP {pr_resp.status_code}): {pr_resp.text}",
                    "code": "GITHUB_AUTH_ERROR"
                }
            elif pr_resp.status_code != 200:
                return {
                    "status": "error",
                    "error": f"GitHub API error fetching PR #{pr_number} (HTTP {pr_resp.status_code}): {pr_resp.text}",
                    "code": "GITHUB_API_ERROR"
                }

            pr_data = pr_resp.json()

            # Fetch PR diff
            diff_headers = self._headers(accept="application/vnd.github.v3.diff")
            diff_resp = await client.get(pr_url, headers=diff_headers)
            diff_text = diff_resp.text if diff_resp.status_code == 200 else ""

            return {
                "status": "success",
                "pr_number": pr_number,
                "title": pr_data.get("title", ""),
                "state": pr_data.get("state", ""),
                "base_branch": pr_data.get("base", {}).get("ref", ""),
                "head_sha": pr_data.get("head", {}).get("sha", ""),
                "diff": diff_text
            }

    async def get_file_content(self, file_path: str, ref: str) -> Dict[str, Any]:
        """Fetch and base64-decode a file from GitHub at a specific ref."""
        if not self.repo_full_name:
            return {
                "status": "error",
                "error": "GITHUB_REPO_FULL_NAME environment variable is not configured.",
                "code": "CONFIG_ERROR"
            }

        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/repos/{self.repo_full_name}/contents/{file_path}?ref={ref}"
            
            try:
                resp = await client.get(url, headers=self._headers())
            except httpx.RequestError as e:
                return {
                    "status": "error",
                    "error": f"Network error connecting to GitHub: {str(e)}",
                    "code": "NETWORK_ERROR"
                }

            if resp.status_code == 404:
                return {
                    "status": "error",
                    "error": f"File '{file_path}' not found at ref '{ref}'.",
                    "code": "FILE_NOT_FOUND"
                }
            elif resp.status_code in (401, 403):
                return {
                    "status": "error",
                    "error": f"GitHub authentication or permission failure (HTTP {resp.status_code}): {resp.text}",
                    "code": "GITHUB_AUTH_ERROR"
                }
            elif resp.status_code != 200:
                return {
                    "status": "error",
                    "error": f"GitHub API error fetching file '{file_path}' (HTTP {resp.status_code}): {resp.text}",
                    "code": "GITHUB_API_ERROR"
                }

            data = resp.json()
            file_sha = data.get("sha", "")
            raw_content = data.get("content", "")
            encoding = data.get("encoding", "")

            if encoding == "base64" and raw_content:
                try:
                    decoded_bytes = base64.b64decode(raw_content)
                    decoded_content = decoded_bytes.decode("utf-8", errors="replace")
                except Exception as e:
                    return {
                        "status": "error",
                        "error": f"Failed to decode base64 content for '{file_path}': {str(e)}",
                        "code": "DECODE_ERROR"
                    }
            else:
                decoded_content = raw_content

            return {
                "status": "success",
                "file_path": file_path,
                "ref": ref,
                "file_sha": file_sha,
                "content": decoded_content
            }

    async def post_review_comment(self, pr_number: int, file_path: str, line: int, body: str) -> Dict[str, Any]:
        """Validate diff line and post a PR review comment."""
        # 1. Fetch PR details & diff
        pr_result = await self.get_pr_details_and_diff(pr_number)
        if pr_result.get("status") == "error":
            return pr_result

        head_sha = pr_result["head_sha"]
        diff_text = pr_result.get("diff", "")

        # 2. Parse diff and validate line position
        added_lines = parse_added_lines_from_diff(diff_text, file_path)
        if not added_lines:
            return {
                "status": "error",
                "error": f"File '{file_path}' has no added or modified lines in PR #{pr_number}.",
                "code": "FILE_NOT_IN_DIFF"
            }

        if line not in added_lines:
            # Attempt to auto-adjust to closest valid added line within +/- 2 lines
            closest_line = min(added_lines, key=lambda l: abs(l - line))
            if abs(closest_line - line) <= 2:
                logger.info(f"Auto-adjusted comment line from {line} to nearest added line {closest_line} in '{file_path}'.")
                line = closest_line
            else:
                return {
                    "status": "error",
                    "error": f"Line {line} in file '{file_path}' is not an added or modified line in PR #{pr_number}. Review comments can only be posted on added or modified lines.",
                    "code": "INVALID_COMMENT_POSITION",
                    "valid_added_lines": sorted(list(added_lines))
                }


        # 3. Post comment via GitHub PR Review Comments API
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/repos/{self.repo_full_name}/pulls/{pr_number}/comments"
            payload = {
                "body": body,
                "commit_id": head_sha,
                "path": file_path,
                "line": line,
                "side": "RIGHT"
            }
            try:
                resp = await client.post(url, headers=self._headers(), json=payload)
            except httpx.RequestError as e:
                return {
                    "status": "error",
                    "error": f"Network error connecting to GitHub: {str(e)}",
                    "code": "NETWORK_ERROR"
                }

            if resp.status_code == 201:
                comment_data = resp.json()
                return {
                    "status": "success",
                    "comment_id": comment_data.get("id"),
                    "html_url": comment_data.get("html_url"),
                    "pr_number": pr_number,
                    "file_path": file_path,
                    "line": line,
                    "body": body
                }
            else:
                return {
                    "status": "error",
                    "error": f"GitHub API failed to post review comment (HTTP {resp.status_code}): {resp.text}",
                    "code": "COMMENT_POST_FAILED"
                }


def parse_added_lines_from_diff(diff_text: str, target_file_path: str) -> Set[int]:
    """Parse unified diff and extract added/modified line numbers for a target file."""
    added_lines: Set[int] = set()
    if not diff_text:
        return added_lines

    file_diffs = re.split(r'^diff --git ', diff_text, flags=re.MULTILINE)

    for file_diff in file_diffs:
        if not file_diff.strip():
            continue

        header_match = re.search(r'^\+\+\+\s+b/(.+)$', file_diff, flags=re.MULTILINE)
        if not header_match:
            continue

        file_path = header_match.group(1).strip()
        if file_path != target_file_path:
            continue

        lines = file_diff.splitlines()
        current_new_line = 0
        in_hunk = False

        for l in lines:
            hunk_match = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', l)
            if hunk_match:
                current_new_line = int(hunk_match.group(1))
                in_hunk = True
                continue

            if in_hunk:
                if l.startswith('+'):
                    added_lines.add(current_new_line)
                    current_new_line += 1
                elif l.startswith('-'):
                    pass
                elif l.startswith(' '):
                    current_new_line += 1
                elif l.startswith('\\'):
                    pass
                else:
                    in_hunk = False

    return added_lines


github_service = GitHubService()


# MCP Tools Registration

@mcp_server.tool()
async def get_diff(pr_number: int) -> Dict[str, Any]:
    """
    Fetch the Pull Request diff and metadata from GitHub.
    
    Args:
        pr_number: The Pull Request number.
    """
    try:
        return await github_service.get_pr_details_and_diff(pr_number)
    except Exception as e:
        logger.error(f"Error in get_diff({pr_number}): {e}")
        return {
            "status": "error",
            "error": f"Internal error fetching PR diff: {str(e)}",
            "code": "INTERNAL_ERROR"
        }


@mcp_server.tool()
async def get_file_context(file_path: str, ref: str) -> Dict[str, Any]:
    """
    Fetch a file from GitHub at a specified commit or branch ref and decode its content.
    
    Args:
        file_path: The file path in the repository.
        ref: The commit SHA, branch name, or tag.
    """
    try:
        return await github_service.get_file_content(file_path, ref)
    except Exception as e:
        logger.error(f"Error in get_file_context({file_path}, {ref}): {e}")
        return {
            "status": "error",
            "error": f"Internal error fetching file context: {str(e)}",
            "code": "INTERNAL_ERROR"
        }


@mcp_server.tool()
async def search_codebase(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Perform a vector similarity search across indexed code chunks in the codebase.
    
    Args:
        query: Search query string.
        top_k: Maximum number of top matching chunks to return (default 5).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error": "GEMINI_API_KEY environment variable is not configured.",
            "code": "CONFIG_ERROR"
        }

    try:
        # Generate query embedding using gemini-embedding-001 with outputDimensionality=384
        embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={api_key}"
        payload = {
            "model": f"models/{EMBEDDING_MODEL}",
            "content": {"parts": [{"text": query}]},
            "outputDimensionality": VECTOR_DIMENSION
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(embed_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30.0)
            except httpx.RequestError as e:
                return {
                    "status": "error",
                    "error": f"Network error connecting to Gemini API: {str(e)}",
                    "code": "NETWORK_ERROR"
                }

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "error": f"Gemini Embedding API error (HTTP {resp.status_code}): {resp.text}",
                    "code": "EMBEDDING_API_ERROR"
                }
            embedding = resp.json().get("embedding", {}).get("values", [])

        if not embedding:
            return {
                "status": "error",
                "error": "Failed to generate query embedding vector.",
                "code": "EMBEDDING_FAILED"
            }

        vec_str = "[" + ",".join(map(str, embedding)) + "]"

        # Perform cosine similarity search via pgvector
        pool = await get_db_pool()
        conn = await pool.acquire()
        try:
            rows = await conn.fetch(
                """
                SELECT file_path, chunk_type, content, (1 - (embedding <=> $1::vector)) AS similarity_score
                FROM code_chunks
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_str,
                top_k,
            )
        finally:
            await pool.release(conn)


        if not rows:
            return {
                "status": "error",
                "error": "No matching code chunks found in database.",
                "code": "EMPTY_SEARCH_RESULTS",
                "results": []
            }

        results = [
            {
                "file_path": r["file_path"],
                "chunk_type": r["chunk_type"],
                "similarity_score": round(float(r["similarity_score"]), 4),
                "content": r["content"],
            }
            for r in rows
        ]

        return {
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in search_codebase('{query}'): {e}")
        return {
            "status": "error",
            "error": f"Database search error: {str(e)}",
            "code": "DATABASE_ERROR"
        }


@mcp_server.tool()
async def post_review_comment(pr_number: int, file_path: str, line: int, body: str) -> Dict[str, Any]:
    """
    Post a line-level review comment on a Pull Request after validating that the line was added/modified.
    
    Args:
        pr_number: The Pull Request number.
        file_path: The relative file path in the repository.
        line: The line number in the new version of the file.
        body: The markdown body of the comment.
    """
    try:
        return await github_service.post_review_comment(pr_number, file_path, line, body)
    except Exception as e:
        logger.error(f"Error in post_review_comment(PR #{pr_number}, {file_path}:{line}): {e}")
        return {
            "status": "error",
            "error": f"Internal error posting review comment: {str(e)}",
            "code": "INTERNAL_ERROR"
        }
