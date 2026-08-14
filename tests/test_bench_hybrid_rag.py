"""Tests for the HybridRAG baseline.

The properties asserted here are the ones that make a baseline number
trustworthy: it never exceeds the shared token budget, it never reads a message
written after the question was asked, it never returns an empty context over a
non-empty store, and it reports its real storage cost.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from bench.adapters.schema import Message as AdapterMessage
from bench.systems.hybrid_rag import HybridRAG
from palimpsest.render import count_tokens
from palimpsest.types import Message

BASE = datetime(2024, 3, 1, 9, 0)

# A hand-made episode. "Zanzibar" and "peregrine" are rare tokens so BM25 has an
# unambiguous target; the paraphrase questions exercise the dense half.
TURNS: list[tuple[int, str, str]] = [
    (0, "Caroline", "I finally adopted a rescue greyhound and named her Juno."),
    (1, "Melanie", "That is wonderful. How is she settling into the apartment?"),
    (2, "Caroline", "She sleeps all day. I take her running along the river at dawn."),
    (3, "Melanie", "I just started a new job as a pastry chef at the hotel downtown."),
    (4, "Caroline", "Congratulations! You were teaching piano lessons before that."),
    (5, "Melanie", "I was. The teaching paid badly so I went back to culinary school."),
    (6, "Caroline", "My apartment lease is up in June and I am moving to Portland."),
    (7, "Melanie", "Portland is lovely. Do you have somewhere to live lined up?"),
    (8, "Caroline", "Not yet. I am flying out next month to look at neighbourhoods."),
    (9, "Melanie", "Bring Juno with you, dogs are a good excuse to meet the neighbours."),
]

FUTURE_TEXT = "I spent my honeymoon in Zanzibar watching a peregrine falcon hunt."
FUTURE_AT = BASE + timedelta(days=30)


def make_messages() -> list[Message]:
    msgs = [
        Message(
            session_id="S1",
            speaker=speaker,
            text=text,
            timestamp=BASE + timedelta(minutes=i),
            msg_id=f"D1:{i}",
        )
        for i, speaker, text in TURNS
    ]
    msgs.append(
        Message(
            session_id="S2",
            speaker="Caroline",
            text=FUTURE_TEXT,
            timestamp=FUTURE_AT,
            msg_id="D2:0",
        )
    )
    return msgs


@pytest.fixture
def system() -> HybridRAG:
    sys = HybridRAG()
    sys.build(make_messages(), [])
    return sys


def test_build_indexes_every_message(system: HybridRAG) -> None:
    assert len(system.index) == len(TURNS) + 1  # short turns, one chunk each
    assert system.stats()["n_messages"] == len(TURNS) + 1
    assert system.stats()["n_chunks"] == len(TURNS) + 1


def test_context_is_non_empty_for_a_reasonable_question(system: HybridRAG) -> None:
    result = system.query(
        "What did Caroline name her dog?", asked_at=None, token_budget=512
    )
    assert result.context.strip()
    assert "Juno" in result.context
    assert result.n_tokens > 0
    assert result.latency_ms > 0


def test_context_is_non_empty_for_a_paraphrased_question(system: HybridRAG) -> None:
    """A non-empty store must never answer with an empty context."""
    result = system.query(
        "Where is she relocating to?", asked_at=None, token_budget=512
    )
    assert result.context.strip()
    assert result.meta["n_rendered"] > 0


@pytest.mark.parametrize("budget", [16, 40, 80, 200, 4096])
def test_query_never_exceeds_the_token_budget(system: HybridRAG, budget: int) -> None:
    result = system.query(
        "Tell me everything about the apartment and the new job",
        asked_at=None,
        token_budget=budget,
    )
    assert count_tokens(result.context) <= budget
    assert result.n_tokens == count_tokens(result.context)
    assert result.n_tokens <= budget


def test_tiny_budget_yields_empty_context_rather_than_an_overflow(
    system: HybridRAG,
) -> None:
    result = system.query("What is her dog called?", asked_at=None, token_budget=2)
    assert result.context == ""
    assert result.n_tokens == 0


def test_an_oversized_chunk_is_truncated_not_dropped() -> None:
    """A non-empty store must never return an empty context, so the best
    candidate gets cut to fit rather than skipped when nothing fits."""
    sys = HybridRAG(max_chunk_tokens=None)
    sys.build(
        [
            Message(
                session_id="S1",
                speaker="Caroline",
                text="marzipan " * 400,
                timestamp=BASE,
                msg_id="D1:0",
            )
        ],
        [],
    )
    result = sys.query("marzipan", asked_at=None, token_budget=64)
    assert result.context.strip()
    assert result.n_tokens <= 64
    assert result.meta["truncated"] is True
    assert result.meta["n_rendered"] == 1


def test_a_chunk_that_does_not_fit_does_not_end_the_scan() -> None:
    """A long top hit must not starve the shorter hits ranked behind it."""
    msgs = [
        Message(
            session_id="S1",
            speaker="Caroline",
            text="marzipan " * 300,
            timestamp=BASE,
            msg_id="D1:0",
        ),
        Message(
            session_id="S1",
            speaker="Melanie",
            text="The marzipan came from the bakery on Alder Street.",
            timestamp=BASE + timedelta(minutes=1),
            msg_id="D1:1",
        ),
    ]
    sys = HybridRAG(max_chunk_tokens=None)
    sys.build(msgs, [])
    result = sys.query("Where did the marzipan come from?", asked_at=None, token_budget=64)
    assert "Alder Street" in result.context
    assert result.meta["truncated"] is False
    assert result.n_tokens <= 64


def test_long_messages_are_split_into_sentence_aligned_chunks() -> None:
    long_text = " ".join(f"Sentence number {i} about the greenhouse." for i in range(80))
    sys = HybridRAG(max_chunk_tokens=64)
    sys.build(
        [
            Message(
                session_id="S1",
                speaker="Caroline",
                text=long_text,
                timestamp=BASE,
                msg_id="D1:0",
            )
        ],
        [],
    )
    assert sys.stats()["n_chunks"] > 5
    assert len(sys.index) == sys.stats()["n_chunks"]
    for _, piece in sys._chunks:
        assert count_tokens(piece) <= 64 + 16  # one sentence may overshoot slightly
    # Chunking is a view over the same text, not a second copy of it.
    unchunked = HybridRAG(max_chunk_tokens=None)
    unchunked.build([Message("S1", "Caroline", long_text, BASE, "D1:0")], [])
    assert sys.stats()["source_text_bytes"] == unchunked.stats()["source_text_bytes"]
    # Every chunk still renders with its parent's speaker and date.
    result = sys.query("greenhouse", asked_at=None, token_budget=256)
    assert result.meta["msg_ids"] == ["D1:0"]
    assert "[2024-03-01] Caroline:" in result.context


def test_asked_at_filters_messages_from_the_future(system: HybridRAG) -> None:
    """The future message is the single best lexical match; the cutoff must still
    hide it, and must stop hiding it once the question is asked later."""
    question = "What happened on the honeymoon in Zanzibar with the peregrine falcon?"

    before = system.query(
        question, asked_at=BASE + timedelta(hours=1), token_budget=512
    )
    assert "Zanzibar" not in before.context
    assert "peregrine" not in before.context
    assert "D2:0" not in before.meta["msg_ids"]
    assert before.meta["prefiltered"] is True

    after = system.query(
        question, asked_at=FUTURE_AT + timedelta(days=1), token_budget=512
    )
    assert "Zanzibar" in after.context
    assert "D2:0" in after.meta["msg_ids"]


def test_cutoff_is_inclusive_of_the_exact_ask_time(system: HybridRAG) -> None:
    result = system.query("Zanzibar peregrine", asked_at=FUTURE_AT, token_budget=512)
    assert "Zanzibar" in result.context


def test_cutoff_before_the_episode_returns_nothing_and_says_so(system: HybridRAG) -> None:
    """LongMemEval has questions dated before the sessions that answer them.
    The cutoff must still hold; the condition is flagged, not papered over."""
    result = system.query(
        "What did Caroline name her dog?",
        asked_at=BASE - timedelta(days=1),
        token_budget=512,
    )
    assert result.context == ""
    assert result.meta["n_visible"] == 0
    assert result.meta["n_visible_messages"] == 0
    assert result.meta["cutoff_hides_all"] is True


def test_cutoff_flag_is_off_when_history_is_visible(system: HybridRAG) -> None:
    result = system.query(
        "What did Caroline name her dog?",
        asked_at=BASE + timedelta(hours=1),
        token_budget=512,
    )
    assert result.meta["cutoff_hides_all"] is False
    assert result.meta["n_visible_messages"] == len(TURNS)


def test_rendered_lines_carry_the_message_date(system: HybridRAG) -> None:
    """Disclosed configuration: excerpts are stamped, matching the engine's own
    render_context, so temporal questions are answerable from the context."""
    result = system.query("What did Caroline name her dog?", asked_at=None, token_budget=512)
    assert "[2024-03-01]" in result.context
    assert "Caroline:" in result.context


def test_timestamps_can_be_disabled() -> None:
    sys = HybridRAG(with_timestamps=False)
    sys.build(make_messages(), [])
    result = sys.query("What did Caroline name her dog?", asked_at=None, token_budget=512)
    assert "Juno" in result.context
    assert "[2024-03-01]" not in result.context


def test_stats_reports_real_storage(system: HybridRAG) -> None:
    stats = system.stats()
    assert stats["bytes_stored"] > 0
    assert stats["source_text_bytes"] > 0
    assert stats["lexical_index_bytes"] > 0
    assert stats["dense_index_bytes"] > 0
    assert (
        stats["bytes_stored"]
        == stats["source_text_bytes"] + stats["dense_index_bytes"] + stats["lexical_index_bytes"]
    )
    # Keeping every message means the store is at least transcript-sized.
    transcript = sum(len(text.encode("utf-8")) for _, _, text in TURNS)
    assert stats["source_text_bytes"] > transcript


def test_binary_quantization_is_smaller_than_float(system: HybridRAG) -> None:
    fp32 = HybridRAG(quantize=False)
    fp32.build(make_messages(), [])
    assert fp32.stats()["dense_index_bytes"] > system.stats()["dense_index_bytes"] * 8


def test_results_are_deterministic(system: HybridRAG) -> None:
    q = "Who changed careers and what do they do now?"
    first = system.query(q, asked_at=None, token_budget=256)
    second = system.query(q, asked_at=None, token_budget=256)
    assert first.context == second.context
    assert first.meta["msg_ids"] == second.meta["msg_ids"]


def test_rebuild_replaces_the_previous_episode(system: HybridRAG) -> None:
    system.build(
        [
            Message(
                session_id="S9",
                speaker="Ana",
                text="I brew kombucha in the garage every weekend.",
                timestamp=BASE,
                msg_id="D9:0",
            )
        ],
        [],
    )
    assert len(system.index) == 1
    result = system.query("What is her hobby?", asked_at=None, token_budget=256)
    assert "kombucha" in result.context
    assert "Juno" not in result.context


def test_accepts_adapter_messages_and_can_index_image_captions() -> None:
    """The bench loaders emit their own Message type, which also carries the
    LoCoMo photo caption. Captions are OFF by default so the baseline sees the
    same text the engine's ingest sees."""
    msgs = [
        AdapterMessage(
            session_id="S1",
            speaker="Caroline",
            text="Look at this!",
            timestamp=BASE,
            msg_id="D1:0",
            image_caption="a chocolate labrador puppy asleep on a red sofa",
        )
    ]
    off = HybridRAG()
    off.build(msgs, [])
    assert "labrador" not in off.query("puppy", asked_at=None, token_budget=256).context

    on = HybridRAG(include_image_captions=True)
    on.build(msgs, [])
    result = on.query("labrador puppy", asked_at=None, token_budget=256)
    assert "labrador" in result.context
    assert on.stats()["config"]["include_image_captions"] is True
