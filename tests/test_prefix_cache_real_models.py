# SPDX-License-Identifier: Apache-2.0
"""End-to-end prefix-cache behaviour, driven through the real scheduler.

The unit tests in ``test_prefix_cache_untrimmable.py`` pin the predicates and
the copy semantics directly. They cannot show that a *stored entry actually
reproduces a cold run*, because that depends on the scheduler, the batch
generator and the model agreeing about how many tokens the entry represents.
Any fake that answered those questions would be asserting this file's own
assumptions back at it.

So these run small real models and compare against cold generation:

- a hybrid model (``ArraysCache`` + ``KVCache`` per layer) for the recurrent
  path, where a key one token short corrupts cumulative state rather than
  merely wasting a prefill;
- a plain ``KVCache`` model for the trimmable path, where the first-response
  snapshot must not run at all — an ``N+1`` entry there evicts the safe ``N``
  one and an identical prompt then matches a supersequence, whereupon the
  scheduler replays ``prompt[-1]`` on a cache that already holds it.

The models are small enough to download once and are skipped when absent, so
this file does not turn an offline checkout into a failure.
"""

import os

import pytest

mx = pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

# Hybrid: 10 ArraysCache + 6 KVCache layers at 350M/8bit.
HYBRID_MODEL = os.environ.get(
    "VLLM_MLX_TEST_HYBRID_MODEL", "LiquidAI/LFM2.5-350M-MLX-8bit"
)
# Plain attention: every layer a KVCache.
KV_MODEL = os.environ.get("VLLM_MLX_TEST_KV_MODEL", "mlx-community/Qwen3-0.6B-8bit")

# Long enough to clear MemoryCacheConfig.min_prefix_tokens (128). A short
# prompt is never stored at all, which makes every comparison below pass
# vacuously — see test_harness_actually_generates_and_caches.
PROMPT = (
    "You are reviewing a changelog. "
    + " ".join(
        f"Entry {i}: the scheduler now records cache coverage explicitly."
        for i in range(24)
    )
    + " Summarise the changelog in three short lines."
)


def _load_or_skip(model_id):
    """Skip rather than fail when the model is not available locally."""
    from mlx_lm.utils import load

    try:
        return load(model_id)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"{model_id} unavailable: {type(exc).__name__}: {exc}")


def _cache_types(model):
    from mlx_lm.models.cache import make_prompt_cache

    return {type(layer).__name__ for layer in make_prompt_cache(model)}


class _Harness:
    """Drives BatchedEngine and exposes what the prefix cache actually holds."""

    def __init__(self, model_id):
        from vllm_mlx.engine.batched import BatchedEngine
        from vllm_mlx.scheduler import SchedulerConfig

        self.engine = BatchedEngine(
            model_id,
            scheduler_config=SchedulerConfig(enable_prefix_cache=True),
        )

    async def start(self):
        await self.engine.start()
        return self

    async def stop(self):
        await self.engine.stop()

    @property
    def _cache(self):
        return self.engine._engine.engine.scheduler.memory_aware_cache

    def entries(self):
        """(key length, key tuple) for every stored entry."""
        cache = self._cache
        if cache is None:
            return []
        return sorted((len(k), k) for k in cache._entries)

    def clear(self):
        cache = self._cache
        if cache is not None:
            cache._entries.clear()
            cache._sorted_keys = []

    async def generate(self, prompt, max_tokens=24):
        out = []
        async for chunk in self.engine.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=0.0
        ):
            out.append(chunk)
        return out[-1].text if out else ""

    def prompt_len(self, prompt):
        tok = self.engine._tokenizer
        return len(tok.encode(prompt))


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


class TestHybridRecurrentModel:
    """Case 1: a stored entry must represent exactly the tokens it is keyed by.

    For a recurrent layer the failure is not a wasted prefill. Replaying a
    token that is already folded into cumulative state changes the state, so
    the continuation diverges from a cold run — silently, and only on the
    second turn.
    """

    @pytest.mark.anyio
    async def test_next_turn_continuation_matches_a_cold_run(self):
        model, _ = _load_or_skip(HYBRID_MODEL)
        types = _cache_types(model)
        assert "ArraysCache" in types, (
            f"{HYBRID_MODEL} no longer has recurrent layers ({types}); this "
            "test would silently stop covering the recurrent path"
        )
        del model

        warm = await _Harness(HYBRID_MODEL).start()
        try:
            first = await warm.generate(PROMPT)
            next_prompt = PROMPT + first
            warm_continuation = await warm.generate(next_prompt)

            stored = warm.entries()

            warm.clear()
            cold_continuation = await warm.generate(next_prompt)
            # Measured while the engine is alive; stop() releases the tokenizer.
            prompt_tokens = warm.prompt_len(PROMPT)
        finally:
            await warm.stop()

        assert warm_continuation == cold_continuation, (
            "the second turn diverged from a cold run, so the reused entry did "
            "not represent the tokens it was keyed by"
        )

        # Coverage is asserted explicitly, not inferred from the output: a
        # future change could reintroduce prompt-only storage of
        # prompt+generated state and still produce matching text on a short
        # prompt.
        for length, _key in stored:
            assert length != prompt_tokens, (
                f"an entry is keyed by exactly the {prompt_tokens} prompt "
                "tokens; for a recurrent cache the stored state also contains "
                "the first generated token, so this key is one token short"
            )


class TestPlainAttentionModel:
    """Case 2: the first-response snapshot must not run for trimmable caches."""

    @pytest.mark.anyio
    async def test_identical_prompt_reuse_matches_cold_and_leaves_one_entry(self):
        model, _ = _load_or_skip(KV_MODEL)
        types = _cache_types(model)
        assert types == {"KVCache"}, (
            f"{KV_MODEL} is no longer plain attention ({types}); this test "
            "would stop covering the trimmable path"
        )
        del model

        h = await _Harness(KV_MODEL).start()
        try:
            h.clear()
            cold = await h.generate(PROMPT)
            after_cold = h.entries()

            warm = await h.generate(PROMPT)
            after_warm = h.entries()
            n = h.prompt_len(PROMPT)
        finally:
            await h.stop()

        assert warm == cold, (
            "an identical prompt produced different output from the cold run; "
            "a supersequence hit replaying prompt[-1] duplicates that token"
        )

        # The completion path legitimately stores prompt+output for a trimmable
        # cache; that entry is reusable by trimming and is the "useful safe
        # entry". What must not exist is the first-response snapshot, which is
        # keyed at exactly N+1 and would evict it.
        lengths = [ln for ln, _ in after_cold]
        assert n + 1 not in lengths, (
            f"prompt is {n} tokens and an entry of exactly {n + 1} was stored: "
            "that is the first-response snapshot, which must be gated to "
            f"non-trimmable topologies. Entries: {lengths}"
        )
        assert all(
            ln == n or ln > n + 1 for ln in lengths
        ), f"unexpected entry lengths for a {n}-token prompt: {lengths}"
        assert len(after_warm) <= 2, (
            f"entries accumulated across identical prompts: "
            f"{[ln for ln, _ in after_warm]}"
        )


def test_harness_actually_generates_and_caches():
    """Guard against the harness passing because nothing ran.

    Both cases above compare outputs; if generation silently produced empty
    strings, or the prefix cache were disabled, they would compare "" to ""
    and pass. This asserts the pieces they depend on are live.
    """
    import asyncio

    _load_or_skip(KV_MODEL)

    async def scenario():
        h = await _Harness(KV_MODEL).start()
        try:
            assert h._cache is not None, "prefix cache is not enabled"
            text = await h.generate(PROMPT)
            return text, h.entries(), h.prompt_len(PROMPT)
        finally:
            await h.stop()

    text, entries, n = asyncio.run(scenario())

    assert len(text.strip()) > 5, f"generation produced {text!r}"
    assert n > 3, f"prompt tokenised to {n} tokens"
    assert entries, "nothing was stored in the prefix cache at all"
