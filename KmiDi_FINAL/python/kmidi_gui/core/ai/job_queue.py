"""AI job queue with caching and persistence.

Jobs are queued, cached, and can be resumed.
Never re-runs AI on identical inputs.
"""

import hashlib
import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..models import AIJob, JobStatus

logger = logging.getLogger(__name__)


class JobQueue:
    """AI job queue with SQLite persistence and caching."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize job queue.
        
        Args:
            db_path: Path to SQLite database (default: ~/.kmidi/jobs.db)
        """
        if db_path is None:
            db_path = Path.home() / ".kmidi" / "jobs.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_input_hash ON ai_jobs(input_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON ai_jobs(status)
            """)
            
            conn.commit()
        logger.info(f"Job queue initialized: {self.db_path}")
    
    def cache_key(self, job_type: str, inputs: Dict[str, Any]) -> str:
        """Generate cache key from job type and inputs.
        
        Args:
            job_type: Type of AI job
            inputs: Input data dictionary
            
        Returns:
            SHA256 hash string
        """
        # Sort inputs for consistent hashing
        sorted_inputs = json.dumps(inputs, sort_keys=True, default=str)
        data = f"{job_type}:{sorted_inputs}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def enqueue(self, job: AIJob) -> bool:
        """Enqueue a job.
        
        Args:
            job: AI job to enqueue
            
        Returns:
            True if enqueued, False if duplicate (cached)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if result already exists (cache hit)
            cursor.execute("""
                SELECT id, result_path FROM ai_jobs
                WHERE input_hash = ? AND status = ?
            """, (job.input_hash, JobStatus.DONE.value))
            
            existing = cursor.fetchone()
            if existing:
                result_path = existing[1]
                if result_path and Path(result_path).exists():
                    logger.info(f"Cache hit for job {job.id}")
                    return False
            
            # Insert new job
            cursor.execute("""
                INSERT OR REPLACE INTO ai_jobs
                (id, type, input_hash, status, result_path, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id,
                job.type,
                job.input_hash,
                job.status.value,
                job.result_path,
                job.error,
                job.created_at.isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
        logger.info(f"Job enqueued: {job.id}")
        return True
    
    def get_job(self, job_id: str) -> Optional[AIJob]:
        """Get job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            AIJob or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, type, input_hash, status, result_path, error, created_at
                FROM ai_jobs WHERE id = ?
            """, (job_id,))
            
            row = cursor.fetchone()
            
            if row:
                return AIJob(
                    id=row[0],
                    type=row[1],
                    input_hash=row[2],
                    status=JobStatus(row[3]),
                    result_path=row[4],
                    error=row[5],
                    created_at=datetime.fromisoformat(row[6])
                )
        return None
    
    def update_job(self, job: AIJob):
        """Update job status.
        
        Args:
            job: Updated job
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ai_jobs
                SET status = ?, result_path = ?, error = ?, updated_at = ?
                WHERE id = ?
            """, (
                job.status.value,
                job.result_path,
                job.error,
                datetime.now().isoformat(),
                job.id
            ))
            
            conn.commit()
        logger.info(f"Job updated: {job.id} -> {job.status.value}")
    
    def get_pending_jobs(self) -> List[AIJob]:
        """Get all pending jobs.
        
        Returns:
            List of pending AIJob objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, type, input_hash, status, result_path, error, created_at
                FROM ai_jobs
                WHERE status = ?
                ORDER BY created_at ASC
            """, (JobStatus.PENDING.value,))
            
            rows = cursor.fetchall()
            
            return [
                AIJob(
                    id=row[0],
                    type=row[1],
                    input_hash=row[2],
                    status=JobStatus(row[3]),
                    result_path=row[4],
                    error=row[5],
                    created_at=datetime.fromisoformat(row[6])
                )
                for row in rows
            ]
    
    def get_cached_result(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached result for input hash.
        
        Args:
            input_hash: Input hash to look up
            
        Returns:
            Cached result dictionary or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT result_path FROM ai_jobs
                WHERE input_hash = ? AND status = ?
            """, (input_hash, JobStatus.DONE.value))
            
            row = cursor.fetchone()
            
            if row and row[0]:
                result_path = Path(row[0])
                if result_path.exists():
                    try:
                        with open(result_path, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except (OSError, json.JSONDecodeError):
                        logger.exception("Failed to load cached result: %s", result_path)
        return None
