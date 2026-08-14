"""Tests for the VectorRAG benchmark baseline.

The load-bearing properties are the ones that would invalidate a benchmark run
if they broke silently: the token budget is a hard ceiling, the ``asked_at``
cutoff really hides the future, retrieval never returns empty over a non-empty
store, and the storage number is measured rather than asserted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bench.adapters.schema import Message as BenchMessage
from bench.systems.vector_rag import VectorRAG
from palimpsest.render import count_tokens
from palimpsest.types import Message

MAY = datetime(2023, 5, 8, 14, 0)
JUN = datetime(2023, 6, 12, 9, 30)
JUL = datetime(2023, 7, 20, 18, 15)


def _msg(session: str, speaker: str, text: str, when: datetime, i: int) -> Message:
    return Message(
        session_id=session,
        speaker=speaker,
        text=text,
        timestamp=when + timedelta(minutes=i),
        msg_id=f"{session}:{i}",
    )


def episode() -> list[Message]:
    """Three sessions, chronological, with one fact that only exists in July."""
    rows = [
        ("S1", "Caroline", "I just started a new job as a paralegal at Hendricks and Sons.", MAY, 0),
        ("S1", "Melanie", "Congratulations! Is the commute from Fremont bad?", MAY, 1),
        ("S1", "Caroline", "It is about forty minutes on the BART, which I do not mind.", MAY, 2),
        ("S1", "Melanie", "That is not terrible. How is the team treating you?", MAY, 3),
        ("S2", "Caroline", "I ran the charity race on Saturday to raise awareness for lupus.", JUN, 0),
        ("S2", "Melanie", "That is wonderful. Did you finish the whole ten kilometres?", JUN, 1),
        ("S2", "Caroline", "I did, in fifty eight minutes, and my knee held up fine.", JUN, 2),
        ("S2", "Melanie", "You should celebrate with that bakery on Telegraph Avenue.", JUN, 3),
        ("S3", "Caroline", "We adopted a golden retriever puppy and named her Waffles.", JUL, 0),
        ("S3", "Melanie", "Waffles is a perfect name for a golden retriever puppy.", JUL, 1),
        ("S3", "Caroline", "She chews everything, so the sofa is already a casualty.", JUL, 2),
    ]
    return [_msg(*row) for row in rows]


@pytest.fixture(scope="module")
def system() -> VectorRAG:
    sys = VectorRAG()
    sys.build(episode(), [])
    return sys


# -- the contract ------------------------------------------------------- #
def test_context_is_non_empty_for_a_reasonable_question(system: VectorRAG):
    result = system.query("What did Caroline run the charity race for?", asked_at=None,
                          token_budget=1024)
    assert result.context.strip()
    assert result.n_tokens > 0
    assert result.meta["n_selected"] > 0


def test_query_never_exceeds_the_token_budget(system: VectorRAG):
    question = "What is the name of Caroline's puppy and where does she work?"
    for budget in (4, 12, 40, 96, 256, 1024, 4096):
        result = system.query(question, asked_at=None, token_budget=budget)
        measured = count_tokens(result.context) if result.context else 0
        assert measured <= budget, f"budget {budget} overrun: {measured} tokens"
        assert result.n_tokens == measured


def test_budget_is_actually_used_when_it_is_generous(system: VectorRAG):
    small = system.query("charity race", asked_at=None, token_budget=64)
    large = system.query("charity race", asked_at=None, token_budget=1024)
    assert large.n_tokens > small.n_tokens
    assert large.meta["n_selected"] > small.meta["n_selected"]


def test_asked_at_cutoff_hides_future_messages(system: VectorRAG):
    question = "What did Caroline name the golden retriever puppy?"

    before = system.query(question, asked_at=JUN.replace(hour=23), token_budget=1024)
    assert "Waffles" not in before.context
    assert "S3" not in "".join(before.meta["msg_ids"])
    assert before.meta["n_hidden_by_cutoff"] > 0

    after = system.query(question, asked_at=JUL.replace(hour=23), token_budget=1024)
    assert "Waffles" in after.context
    assert after.meta["n_hidden_by_cutoff"] == 0


def test_cutoff_drops_chunks_that_straddle_it(system: VectorRAG):
    """No chunk may be served whose last turn is after the cutoff: its embedding
    was computed over that future text, so admitting it would leak into ranking."""
    cutoff = JUL.replace(hour=18, minute=15)  # the first S3 turn, exactly
    result = system.query("puppy", asked_at=cutoff, token_budget=1024)
    for chunk in system.chunks:
        if chunk.end > cutoff:
            assert chunk.text.splitlines()[-1] not in result.context


def test_tz_aware_asked_at_filters_instead_of_crashing(system: VectorRAG):
    """The adapters emit naive local times, but ``QAItem.asked_at`` round-trips
    through ``fromisoformat``, which yields an aware datetime for any offset-
    bearing string. Comparing the two raised TypeError mid-run."""
    question = "What did Caroline name the golden retriever puppy?"
    aware = JUN.replace(hour=23, tzinfo=UTC)

    before = system.query(question, asked_at=aware, token_budget=1024)
    assert "Waffles" not in before.context
    assert before.meta["n_hidden_by_cutoff"] > 0

    naive = system.query(question, asked_at=JUN.replace(hour=23), token_budget=1024)
    assert before.context == naive.context


def test_image_captions_can_be_disabled_and_the_setting_is_reported():
    """The engine under test never sees captions, so the knob has to be visible
    in ``stats()`` rather than baked in — the setting travels with the number."""
    msgs = [
        BenchMessage("S1", "Caroline", "Look at this!", MAY, "S1:0",
                     image_caption="a sunflower field at dusk in Provence"),
        BenchMessage("S1", "Melanie", "Beautiful.", MAY.replace(minute=1), "S1:1"),
    ]
    off = VectorRAG(include_image_captions=False)
    off.build(msgs, [])
    assert off.stats()["include_image_captions"] is False
    assert "sunflower" not in off.query("sunflower field", asked_at=None,
                                        token_budget=512).context

    on = VectorRAG()
    on.build(msgs, [])
    assert on.stats()["include_image_captions"] is True


def test_stats_reports_bytes_stored(system: VectorRAG):
    stats = system.stats()
    assert stats["bytes_stored"] > 0
    assert stats["source_text_bytes"] > 0
    assert stats["index_bytes"] > 0
    assert stats["bytes_stored"] == stats["source_text_bytes"] + stats["index_bytes"]
    assert stats["n_chunks"] > 0
    assert stats["n_messages"] == len(episode())


def test_index_is_binary_quantized_not_fp32(system: VectorRAG):
    """The storage claim only means something if the index really is 1 bit/dim."""
    stats = system.stats()
    assert stats["quantized"] is True
    assert stats["bits_per_dim"] == 1
    dim = stats["embedding_dim"]
    assert stats["index_constant_bytes"] == dim * 4  # the quantization thresholds
    assert stats["index_vector_bytes"] == stats["n_chunks"] * (dim // 8)
    assert stats["bytes_per_vector"] == dim / 8
    assert stats["index_vector_bytes"] * 32 == stats["fp32_index_bytes_equivalent"]


def test_fp32_configuration_reports_the_larger_index():
    """The 32x is a claim about the term that scales, so compare per-vector bytes."""
    fp32 = VectorRAG(quantize=False)
    fp32.build(episode(), [])
    quantized = VectorRAG()
    quantized.build(episode(), [])
    assert fp32.stats()["index_constant_bytes"] == 0
    assert fp32.stats()["bytes_per_vector"] == 32 * quantized.stats()["bytes_per_vector"]
    assert fp32.stats()["index_bytes"] > quantized.stats()["index_vector_bytes"]


def test_chunks_never_span_a_session_or_blow_the_chunk_size(system: VectorRAG):
    assert system.chunks
    for chunk in system.chunks:
        assert chunk.session_id
        assert chunk.start <= chunk.end
        assert chunk.n_tokens <= system.chunk_tokens + 24  # + date header slack
        sessions = {mid.split(":")[0] for mid in chunk.msg_ids}
        assert len(sessions) == 1


def test_long_turns_are_split_into_several_chunks():
    long_turn = " ".join(
        f"Sentence number {i} describes a distinct detail about the quarterly audit."
        for i in range(60)
    )
    sys = VectorRAG()
    sys.build([Message("S1", "user", long_turn, MAY, "S1:0")], [])
    assert len(sys.chunks) > 1
    assert all(c.n_tokens <= sys.chunk_tokens + 24 for c in sys.chunks)


def test_image_captions_from_bench_messages_are_indexed():
    msgs = [
        BenchMessage("S1", "Caroline", "Look at this!", MAY, "S1:0",
                     image_caption="a sunflower field at dusk in Provence"),
        BenchMessage("S1", "Melanie", "Beautiful.", MAY.replace(minute=1), "S1:1"),
    ]
    sys = VectorRAG()
    sys.build(msgs, [])
    result = sys.query("sunflower field", asked_at=None, token_budget=512)
    assert "sunflower field at dusk" in result.context


def test_query_is_deterministic(system: VectorRAG):
    a = system.query("Where does Caroline work?", asked_at=None, token_budget=256)
    b = system.query("Where does Caroline work?", asked_at=None, token_budget=256)
    assert a.context == b.context
    assert a.meta["msg_ids"] == b.meta["msg_ids"]


def test_excerpts_are_rendered_oldest_first(system: VectorRAG):
    result = system.query("Caroline", asked_at=None, token_budget=4096)
    ids = result.meta["msg_ids"]
    assert ids == sorted(ids, key=lambda m: (m.split(":")[0], int(m.split(":")[1])))


def test_latency_is_measured(system: VectorRAG):
    result = system.query("commute", asked_at=None, token_budget=256)
    assert result.latency_ms > 0


def test_empty_store_returns_empty_context_without_crashing():
    sys = VectorRAG()
    sys.build([], [])
    result = sys.query("anything at all", asked_at=None, token_budget=1024)
    assert result.context == ""
    assert result.n_tokens == 0
    assert sys.stats()["bytes_stored"] == 0


def test_query_before_build_is_safe():
    result = VectorRAG().query("anything", asked_at=None, token_budget=1024)
    assert result.context == ""
    assert result.meta["reason"] == "empty_store"
