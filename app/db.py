"""Database setup and session management."""

import os
import logging
from typing import Optional
import asyncpg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connection pooling via asyncpg."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self._pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            if not self.db_url:
                raise ValueError("DATABASE_URL environment variable is not configured.")
            logger.info("Initializing asyncpg connection pool...")
            self._pool = await asyncpg.create_pool(
                dsn=self.db_url,
                min_size=1,
                max_size=10,
            )
        return self._pool

    async def close(self):
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed.")


db_manager = DatabaseManager()


async def get_db_pool() -> asyncpg.Pool:
    """Helper to get asyncpg pool instance."""
    return await db_manager.get_pool()

