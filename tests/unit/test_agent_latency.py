"""
Unit tests for lower-latency local agent workflow features.

Tests connection pooling, response caching, health check TTL,
and exponential backoff retry logic.
"""

import asyncio
import importlib.util
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helper: import crewai module directly to bypass pre-existing async_hub issue
# ---------------------------------------------------------------------------

def _load_crewai():
    spec = importlib.util.spec_from_file_location(
        "crewai_music_agents",
        "music_brain/agents/crewai_music_agents.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_crewai = _load_crewai()
_LRUResponseCache = _crewai._LRUResponseCache
LocalLLM = _crewai.LocalLLM
LocalLLMConfig = _crewai.LocalLLMConfig
OnnxLLM = _crewai.OnnxLLM
OnnxLLMConfig = _crewai.OnnxLLMConfig


# ---------------------------------------------------------------------------
# Orchestrator imports (no pre-existing issues)
# ---------------------------------------------------------------------------

from music_brain.orchestrator.orchestrator import (
    AIOrchestrator,
    OrchestratorConfig,
)
from music_brain.orchestrator.interfaces import (
    ProcessorInterface,
    ProcessorResult,
    ExecutionContext,
)
from music_brain.orchestrator.pipeline import Pipeline


# ============================================================================
# Test _LRUResponseCache
# ============================================================================


@pytest.mark.unit
class TestLRUResponseCache:
    """Tests for the LRU response cache."""

    def test_basic_put_and_get(self):
        cache = _LRUResponseCache(max_size=10)
        cache.put("model", "prompt", "response")
        assert cache.get("model", "prompt") == "response"

    def test_cache_miss_returns_none(self):
        cache = _LRUResponseCache(max_size=10)
        assert cache.get("model", "missing_prompt") is None

    def test_system_prompt_included_in_key(self):
        cache = _LRUResponseCache(max_size=10)
        cache.put("model", "prompt", "resp_no_sys")
        cache.put("model", "prompt", "resp_with_sys", system="You are a composer")
        assert cache.get("model", "prompt") == "resp_no_sys"
        assert cache.get("model", "prompt", "You are a composer") == "resp_with_sys"

    def test_eviction_at_max_size(self):
        cache = _LRUResponseCache(max_size=2)
        cache.put("m", "p1", "r1")
        cache.put("m", "p2", "r2")
        cache.put("m", "p3", "r3")
        assert len(cache) == 2
        assert cache.get("m", "p1") is None  # evicted (LRU)
        assert cache.get("m", "p2") == "r2"
        assert cache.get("m", "p3") == "r3"

    def test_lru_ordering_on_access(self):
        cache = _LRUResponseCache(max_size=2)
        cache.put("m", "p1", "r1")
        cache.put("m", "p2", "r2")
        # Access p1 -> moves to MRU position
        cache.get("m", "p1")
        # Insert p3 -> evicts p2 (now LRU)
        cache.put("m", "p3", "r3")
        assert cache.get("m", "p2") is None
        assert cache.get("m", "p1") == "r1"

    def test_update_existing_entry(self):
        cache = _LRUResponseCache(max_size=10)
        cache.put("m", "p1", "old_response")
        cache.put("m", "p1", "new_response")
        assert cache.get("m", "p1") == "new_response"
        assert len(cache) == 1

    def test_clear(self):
        cache = _LRUResponseCache(max_size=10)
        cache.put("m", "p1", "r1")
        cache.put("m", "p2", "r2")
        cache.clear()
        assert len(cache) == 0
        assert cache.get("m", "p1") is None


# ============================================================================
# Test LocalLLM Connection Pooling and Caching
# ============================================================================


@pytest.mark.unit
class TestLocalLLMLatency:
    """Tests for LocalLLM latency optimizations."""

    def test_uses_requests_session(self):
        """Verify LocalLLM uses a requests.Session for connection pooling."""
        import requests
        llm = LocalLLM()
        assert isinstance(llm._session, requests.Session)
        llm.close()

    def test_health_check_ttl_caching(self):
        """Verify health check result is cached for the TTL period."""
        config = LocalLLMConfig(health_check_ttl=60.0)
        llm = LocalLLM(config)
        initial_time = llm._health_checked_at
        assert initial_time > 0

        # Access is_available again - should NOT re-check (TTL not expired)
        with patch.object(llm._session, 'get') as mock_get:
            _ = llm.is_available
            mock_get.assert_not_called()

        llm.close()

    def test_health_check_refreshes_after_ttl(self):
        """Verify health check refreshes when TTL expires."""
        config = LocalLLMConfig(health_check_ttl=0.0)  # Always re-check
        llm = LocalLLM(config)

        with patch.object(llm._session, 'get') as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            _ = llm.is_available
            mock_get.assert_called_once()

        llm.close()

    def test_generate_caches_response(self):
        """Verify identical generate() calls return cached responses."""
        llm = LocalLLM()
        llm._available = True

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "C Am F G"}

        with patch.object(llm._session, 'post', return_value=mock_resp) as mock_post:
            result1 = llm.generate("Write a progression")
            result2 = llm.generate("Write a progression")

            assert result1 == "C Am F G"
            assert result2 == "C Am F G"
            # Only one actual HTTP call - second is cached
            mock_post.assert_called_once()

        llm.close()

    def test_chat_caches_response(self):
        """Verify identical chat() calls return cached responses."""
        llm = LocalLLM()
        llm._available = True

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "Try Dm7 G7 Cmaj7"}}

        messages = [{"role": "user", "content": "Suggest a jazz progression"}]

        with patch.object(llm._session, 'post', return_value=mock_resp) as mock_post:
            result1 = llm.chat(messages)
            result2 = llm.chat(messages)

            assert result1 == "Try Dm7 G7 Cmaj7"
            assert result2 == "Try Dm7 G7 Cmaj7"
            mock_post.assert_called_once()

        llm.close()

    def test_clear_cache(self):
        """Verify clear_cache() empties the response cache."""
        llm = LocalLLM()
        llm._cache.put("m", "p", "r")
        assert len(llm._cache) == 1
        llm.clear_cache()
        assert len(llm._cache) == 0
        llm.close()

    def test_close_session(self):
        """Verify close() cleans up the HTTP session."""
        llm = LocalLLM()
        with patch.object(llm._session, 'close') as mock_close:
            llm.close()
            mock_close.assert_called_once()


# ============================================================================
# Test OnnxLLM Connection Pooling and Caching
# ============================================================================


@pytest.mark.unit
class TestOnnxLLMLatency:
    """Tests for OnnxLLM latency optimizations."""

    def test_uses_requests_session(self):
        """Verify OnnxLLM uses a requests.Session for connection pooling."""
        import requests
        llm = OnnxLLM()
        assert isinstance(llm._session, requests.Session)
        llm.close()

    def test_generate_caches_response(self):
        """Verify identical generate() calls return cached responses."""
        llm = OnnxLLM()
        llm._available = True

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"output": "Am Em F G"}

        with patch.object(llm._session, 'post', return_value=mock_resp) as mock_post:
            r1 = llm.generate("Write a progression")
            r2 = llm.generate("Write a progression")
            assert r1 == "Am Em F G"
            assert r2 == r1
            mock_post.assert_called_once()

        llm.close()

    def test_chat_caches_response(self):
        """Verify identical chat() calls return cached responses."""
        llm = OnnxLLM()
        llm._available = True

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"output": "Dm7 G7 Cmaj7"}

        messages = [{"role": "user", "content": "Suggest jazz chords"}]

        with patch.object(llm._session, 'post', return_value=mock_resp) as mock_post:
            r1 = llm.chat(messages)
            r2 = llm.chat(messages)
            assert r1 == "Dm7 G7 Cmaj7"
            assert r2 == r1
            mock_post.assert_called_once()

        llm.close()

    def test_health_check_ttl(self):
        """Verify OnnxLLM caches health check for TTL period."""
        config = OnnxLLMConfig(health_check_ttl=60.0)
        llm = OnnxLLM(config)
        initial_time = llm._health_checked_at
        assert initial_time > 0

        with patch.object(llm._session, 'get') as mock_get:
            _ = llm.is_available
            mock_get.assert_not_called()

        llm.close()


# ============================================================================
# Test Orchestrator Exponential Backoff
# ============================================================================


class _FailNTimesProcessor(ProcessorInterface):
    """Test processor that fails N times then succeeds."""

    def __init__(self, fail_count: int):
        super().__init__()
        self._fail_count = fail_count
        self._attempt = 0

    async def process(self, input_data, context):
        self._attempt += 1
        if self._attempt <= self._fail_count:
            raise RuntimeError(f"Simulated failure #{self._attempt}")
        return ProcessorResult(success=True, data=f"ok after {self._attempt} attempts")


@pytest.mark.unit
class TestOrchestratorBackoff:
    """Tests for orchestrator exponential backoff retry logic."""

    def test_config_has_retry_base_delay(self):
        """Verify retry_base_delay exists in config with default."""
        config = OrchestratorConfig()
        assert config.retry_base_delay == 0.5
        assert "retry_base_delay" in config.to_dict()

    def test_backoff_delay_applied_on_retry(self):
        """Verify retries include exponential backoff delays."""
        config = OrchestratorConfig(
            max_retries=2,
            retry_base_delay=0.1,
            default_timeout=5.0,
        )
        orchestrator = AIOrchestrator(config)

        processor = _FailNTimesProcessor(fail_count=2)
        pipeline = Pipeline("test_backoff")
        pipeline.add_stage("flaky", processor)

        async def run():
            start = time.perf_counter()
            result = await orchestrator.execute(pipeline, "test_input")
            elapsed = time.perf_counter() - start
            return result, elapsed

        result, elapsed = asyncio.get_event_loop().run_until_complete(run())

        assert result.success
        assert result.final_output == "ok after 3 attempts"
        # With base_delay=0.1: first retry sleeps 0.1s, second sleeps 0.2s
        # Total backoff >= 0.3s (minus some overhead tolerance)
        assert elapsed >= 0.25, f"Expected backoff delay, got {elapsed:.3f}s"

    def test_successful_first_attempt_no_delay(self):
        """Verify no backoff delay when first attempt succeeds."""
        config = OrchestratorConfig(
            max_retries=2,
            retry_base_delay=1.0,
            default_timeout=5.0,
        )
        orchestrator = AIOrchestrator(config)

        processor = _FailNTimesProcessor(fail_count=0)
        pipeline = Pipeline("test_no_delay")
        pipeline.add_stage("quick", processor)

        async def run():
            start = time.perf_counter()
            result = await orchestrator.execute(pipeline, "test_input")
            elapsed = time.perf_counter() - start
            return result, elapsed

        result, elapsed = asyncio.get_event_loop().run_until_complete(run())

        assert result.success
        # Should complete quickly without any backoff delay
        assert elapsed < 0.5, f"Unexpected delay: {elapsed:.3f}s"

    def test_all_retries_exhausted(self):
        """Verify all retries exhausted returns failure."""
        config = OrchestratorConfig(
            max_retries=1,
            retry_base_delay=0.05,
            default_timeout=5.0,
        )
        orchestrator = AIOrchestrator(config)

        processor = _FailNTimesProcessor(fail_count=10)  # Always fails
        pipeline = Pipeline("test_exhaust")
        pipeline.add_stage("always_fail", processor)

        async def run():
            return await orchestrator.execute(pipeline, "test_input")

        result = asyncio.get_event_loop().run_until_complete(run())
        assert not result.success
        assert "always_fail" in result.error
