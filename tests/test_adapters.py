"""Adapter tests: published counts, total date parsing, chronology, evidence
resolution, and schema round-trips.

The evidence-resolution tests do not merely assert "clean" -- LoCoMo is a
known-dirty dataset, so they pin the EXACT measured miss rate. If the data or the
loader changes, the number moves and the test says by how much.
"""

from __future__ import annotations

import collections
import itertools
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bench.adapters import locomo as locomo_mod
from bench.adapters import longmemeval as lme_mod
from bench.adapters.locomo import load_locomo, repair_evidence_id
from bench.adapters.longmemeval import iter_longmemeval, load_longmemeval
from bench.adapters.schema import (
    CATEGORY_COUNTS,
    LOCOMO_CATEGORY_COUNTS,
    LONGMEMEVAL_CATEGORY_COUNTS,
    PUBLISHED_TOTALS,
    AdapterError,
    DateParseError,
    Episode,
    Message,
    QAItem,
    assign_timestamps,
    snake_case,
)

# Measured on data/raw/locomo10.json (see bench/adapters/REPORT.md).
LOCOMO_EVIDENCE_REFS = 2815
LOCOMO_UNRESOLVED_REFS = 9
LOCOMO_QA_WITH_BAD_EVIDENCE = 9
LOCOMO_EMPTY_EVIDENCE_QA = 4

LME_ABSTENTION_QUESTIONS = 30
LME_EVIDENCE_FALLBACK_QUESTIONS = 21


@pytest.fixture(scope="module")
def locomo() -> list[Episode]:
    if not locomo_mod.LOCOMO_DEFAULT_PATH.exists():
        pytest.skip(f"missing {locomo_mod.LOCOMO_DEFAULT_PATH}")
    return load_locomo()


@pytest.fixture(scope="module")
def lme() -> list[Episode]:
    path = lme_mod.LONGMEMEVAL_PATHS["oracle"]
    if not path.exists():
        pytest.skip(f"missing {path}")
    return load_longmemeval()


# --------------------------------------------------------------------------
# LoCoMo
# --------------------------------------------------------------------------


def test_locomo_totals_match_published(locomo):
    expected = PUBLISHED_TOTALS["locomo"]
    assert len(locomo) == expected["episodes"]
    assert sum(len(e.messages) for e in locomo) == expected["messages"]
    assert sum(len(e.qa) for e in locomo) == expected["qa"]


def test_locomo_category_counts_match_published(locomo):
    counts = collections.Counter(q.category for e in locomo for q in e.qa)
    assert dict(counts) == LOCOMO_CATEGORY_COUNTS
    assert CATEGORY_COUNTS["locomo"] == LOCOMO_CATEGORY_COUNTS


def test_locomo_categories_are_names_not_ints(locomo):
    for episode in locomo:
        for item in episode.qa:
            assert isinstance(item.category, str)
            assert item.category in LOCOMO_CATEGORY_COUNTS


def test_locomo_adversarial_flag_matches_category(locomo):
    for episode in locomo:
        for item in episode.qa:
            assert item.adversarial == (item.category == "adversarial")
    n_adv = sum(1 for e in locomo for q in e.qa if q.adversarial)
    assert n_adv == LOCOMO_CATEGORY_COUNTS["adversarial"]


def test_locomo_every_message_has_a_real_timestamp(locomo):
    for episode in locomo:
        for message in episode.messages:
            assert isinstance(message.timestamp, datetime)
            assert 2000 < message.timestamp.year < 2100


def test_locomo_messages_are_strictly_chronological(locomo):
    for episode in locomo:
        stamps = [m.timestamp for m in episode.messages]
        assert stamps == sorted(stamps), episode.episode_id
        assert all(a < b for a, b in zip(stamps, stamps[1:])), episode.episode_id


def test_locomo_sessions_are_chronological_and_contiguous(locomo):
    """Session blocks must not interleave once sorted by parsed datetime."""
    for episode in locomo:
        order = [m.session_id for m in episode.messages]
        collapsed = [k for k, _ in itertools.groupby(order)]
        assert len(collapsed) == len(set(collapsed)), episode.episode_id
        assert len(collapsed) == episode.meta["n_sessions"]


def test_locomo_msg_ids_unique_within_episode(locomo):
    for episode in locomo:
        ids = [m.msg_id for m in episode.messages]
        assert len(ids) == len(set(ids)), episode.episode_id


def test_locomo_no_empty_message_text(locomo):
    empty = [m.msg_id for e in locomo for m in e.messages if not m.text.strip()]
    assert empty == []


def test_locomo_gold_answers_are_non_empty_strings(locomo):
    for episode in locomo:
        for item in episode.qa:
            assert isinstance(item.gold_answer, str)
            assert item.gold_answer.strip(), item.qid


def test_locomo_asked_at_is_end_of_history(locomo):
    for episode in locomo:
        assert episode.end is not None
        for item in episode.qa:
            assert item.asked_at == episode.end


def test_locomo_evidence_resolution_rate_is_exactly_as_measured(locomo):
    """LoCoMo's evidence annotations are dirty. Pin the rate, do not hide it."""
    total_refs = 0
    bad_refs: list[tuple[str, str]] = []
    qa_with_bad = 0
    for episode in locomo:
        for item in episode.qa:
            total_refs += len(item.evidence_ids)
            missing = episode.unresolved_evidence(item)
            bad_refs.extend((item.qid, m) for m in missing)
            if missing:
                qa_with_bad += 1

    rate = len(bad_refs) / total_refs
    assert total_refs == LOCOMO_EVIDENCE_REFS
    assert len(bad_refs) == LOCOMO_UNRESOLVED_REFS, (
        f"unresolved evidence refs changed: {len(bad_refs)}/{total_refs} "
        f"({rate:.4%}); offenders={bad_refs}"
    )
    assert qa_with_bad == LOCOMO_QA_WITH_BAD_EVIDENCE
    assert rate < 0.005


def test_locomo_non_adversarial_evidence_resolution(locomo):
    """Every non-adversarial QA item's evidence should resolve. 9 do not; the
    failure rate is asserted explicitly so it can never quietly grow."""
    checked = 0
    failed = []
    for episode in locomo:
        for item in episode.qa:
            if item.adversarial:
                continue
            checked += 1
            missing = episode.unresolved_evidence(item)
            if missing:
                failed.append((item.qid, item.category, missing))

    rate = len(failed) / checked
    assert checked == PUBLISHED_TOTALS["locomo"]["qa"] - LOCOMO_CATEGORY_COUNTS["adversarial"]
    assert len(failed) == LOCOMO_QA_WITH_BAD_EVIDENCE, (
        f"non-adversarial evidence failures: {len(failed)}/{checked} "
        f"({rate:.4%}); {failed}"
    )
    assert rate < 0.01


def test_locomo_adversarial_evidence_fully_resolves(locomo):
    failed = [
        item.qid
        for episode in locomo
        for item in episode.qa
        if item.adversarial and episode.unresolved_evidence(item)
    ]
    assert failed == []


def test_locomo_empty_evidence_lists_are_counted(locomo):
    empty = [q.qid for e in locomo for q in e.qa if not q.evidence_ids]
    assert len(empty) == LOCOMO_EMPTY_EVIDENCE_QA
    assert {q.category for e in locomo for q in e.qa if not q.evidence_ids} == {
        "open_domain"
    }


def test_locomo_evidence_repair_separates_formatting_from_real_errors():
    """Repairing the malformed strings leaves exactly the ids that are
    well-formed but point at turns that do not exist -- the genuine annotation
    errors, as opposed to the formatting slips."""
    episodes = load_locomo(repair_evidence=True)
    bad = [
        (item.qid, m)
        for episode in episodes
        for item in episode.qa
        for m in episode.unresolved_evidence(item)
    ]
    assert [m for _, m in bad] == ["D10:19", "D4:36"], bad
    # The bare "D" carries no id at all and is dropped; that item keeps its two
    # other evidence ids, so no QA item loses its evidence entirely.
    voided = [
        item.qid
        for episode in episodes
        for item in episode.qa
        if not item.evidence_ids
    ]
    assert len(voided) == LOCOMO_EMPTY_EVIDENCE_QA


def test_repair_evidence_id_shapes():
    assert repair_evidence_id("D8:6; D9:17") == ["D8:6", "D9:17"]
    assert repair_evidence_id("D9:1 D4:4 D4:6") == ["D9:1", "D4:4", "D4:6"]
    assert repair_evidence_id("D:11:26") == ["D11:26"]
    assert repair_evidence_id("D30:05") == ["D30:5"]
    assert repair_evidence_id("D") == []


def test_locomo_message_by_id_round_trips(locomo):
    episode = locomo[0]
    for message in episode.messages[:50]:
        assert episode.message_by_id(message.msg_id) is message
    assert episode.message_by_id("D999:999") is None


def test_locomo_image_turns_keep_their_caption(locomo):
    captioned = [m for e in locomo for m in e.messages if m.image_caption]
    assert len(captioned) == 1226  # measured: every blip_caption turn
    assert all(isinstance(m.image_urls, tuple) for m in captioned)


def test_locomo_bad_date_raises_with_the_offending_string(tmp_path: Path):
    payload = [
        {
            "sample_id": "conv-x",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "sometime last Tuesday",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}],
            },
            "qa": [],
        }
    ]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DateParseError) as excinfo:
        load_locomo(path)
    assert "sometime last Tuesday" in str(excinfo.value)


def test_locomo_parser_handles_am_pm_and_midnight():
    parse = locomo_mod.parse_locomo_datetime
    assert parse("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56)
    assert parse("12:09 am on 13 September, 2023") == datetime(2023, 9, 13, 0, 9)
    assert parse("12:48 AM on 1 February, 2023") == datetime(2023, 2, 1, 0, 48)
    with pytest.raises(DateParseError):
        parse("")
    with pytest.raises(DateParseError):
        parse(None)


def test_locomo_sessions_are_sorted_by_date_not_session_number(tmp_path: Path):
    """Session numbers are not trusted; sorting is by parsed datetime."""
    payload = [
        {
            "sample_id": "conv-y",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "1:00 pm on 10 May, 2023",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "later"}],
                "session_2_date_time": "1:00 pm on 1 May, 2023",
                "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "earlier"}],
            },
            "qa": [],
        }
    ]
    path = tmp_path / "unsorted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    episode = load_locomo(path)[0]
    assert [m.msg_id for m in episode.messages] == ["D2:1", "D1:1"]
    assert episode.messages[0].timestamp < episode.messages[1].timestamp


def test_locomo_orphan_date_keys_are_recorded_not_fatal(locomo):
    by_id = {e.episode_id: e for e in locomo}
    assert by_id["conv-26"].meta["orphan_date_sessions"] == list(range(20, 36))
    others = [e for e in locomo if e.episode_id != "conv-26"]
    assert all(e.meta["orphan_date_sessions"] == [] for e in others)


def test_locomo_session_without_a_date_raises(tmp_path: Path):
    payload = [
        {
            "sample_id": "conv-z",
            "conversation": {
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}]
            },
            "qa": [],
        }
    ]
    path = tmp_path / "nodate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AdapterError):
        load_locomo(path)


# --------------------------------------------------------------------------
# LongMemEval
# --------------------------------------------------------------------------


def test_lme_totals_match_published(lme):
    expected = PUBLISHED_TOTALS["longmemeval"]
    assert len(lme) == expected["episodes"]
    assert sum(len(e.qa) for e in lme) == expected["qa"]


def test_lme_category_counts_match_published(lme):
    counts = collections.Counter(q.category for e in lme for q in e.qa)
    assert dict(counts) == LONGMEMEVAL_CATEGORY_COUNTS
    assert CATEGORY_COUNTS["longmemeval"] == LONGMEMEVAL_CATEGORY_COUNTS


def test_lme_categories_are_snake_case(lme):
    for episode in lme:
        for item in episode.qa:
            assert item.category == snake_case(episode.meta["question_type_raw"])
            assert "-" not in item.category and " " not in item.category


def test_lme_abstention_variants_are_flagged(lme):
    abstain = [q for e in lme for q in e.qa if q.adversarial]
    assert len(abstain) == LME_ABSTENTION_QUESTIONS
    assert all(q.qid.endswith("_abs") for q in abstain)


def test_lme_all_dates_parse_and_are_real(lme):
    for episode in lme:
        assert episode.qa[0].asked_at is not None
        for message in episode.messages:
            assert isinstance(message.timestamp, datetime)
            assert 2000 < message.timestamp.year < 2100


def test_lme_messages_are_chronological(lme):
    for episode in lme:
        stamps = [m.timestamp for m in episode.messages]
        assert stamps == sorted(stamps), episode.episode_id


def test_lme_msg_ids_unique_within_episode(lme):
    for episode in lme:
        ids = [m.msg_id for m in episode.messages]
        assert len(ids) == len(set(ids)), episode.episode_id


def test_lme_speakers_are_user_or_assistant(lme):
    speakers = {m.speaker for e in lme for m in e.messages}
    assert speakers == {"user", "assistant"}


def test_lme_evidence_ids_all_resolve(lme):
    total = 0
    missing = []
    for episode in lme:
        for item in episode.qa:
            total += len(item.evidence_ids)
            missing.extend(episode.unresolved_evidence(item))
    assert total > 0
    assert missing == [], f"{len(missing)}/{total} LongMemEval evidence ids unresolved"


def test_lme_evidence_falls_back_only_for_abstention_questions(lme):
    fallback = [e for e in lme if e.meta["evidence_source"] == "answer_sessions"]
    assert len(fallback) == LME_EVIDENCE_FALLBACK_QUESTIONS
    assert all(e.episode_id.endswith("_abs") for e in fallback)
    assert all(e.meta["evidence_source"] != "none" for e in lme)


def test_lme_bad_date_raises_with_the_offending_string(tmp_path: Path):
    payload = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "?",
            "answer": "a",
            "question_date": "2023/04/10 (Mon) 23:07",
            "haystack_dates": ["last Thursday"],
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [[{"role": "user", "content": "hi"}]],
            "answer_session_ids": ["s1"],
        }
    ]
    path = tmp_path / "bad_lme.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DateParseError) as excinfo:
        load_longmemeval(path)
    assert "last Thursday" in str(excinfo.value)


def test_lme_parser_formats():
    parse = lme_mod.parse_longmemeval_datetime
    assert parse("2023/04/10 (Mon) 23:07") == datetime(2023, 4, 10, 23, 7)
    assert parse("2023/04/10 23:07") == datetime(2023, 4, 10, 23, 7)
    with pytest.raises(DateParseError):
        parse("2023-04-10 23:07")


def test_lme_sessions_sorted_by_date_not_file_order(tmp_path: Path):
    payload = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "?",
            "answer": "a",
            "question_date": "2023/05/01 (Mon) 10:00",
            "haystack_dates": ["2023/04/10 (Mon) 23:07", "2023/01/02 (Mon) 08:00"],
            "haystack_session_ids": ["late", "early"],
            "haystack_sessions": [
                [{"role": "user", "content": "second"}],
                [{"role": "user", "content": "first"}],
            ],
            "answer_session_ids": ["early"],
        }
    ]
    path = tmp_path / "unsorted_lme.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    episode = load_longmemeval(path)[0]
    assert [m.text for m in episode.messages] == ["first", "second"]
    assert episode.qa[0].evidence_ids == ["early:0"]


def test_lme_parallel_array_mismatch_raises(tmp_path: Path):
    payload = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "?",
            "answer": "a",
            "question_date": "2023/05/01 (Mon) 10:00",
            "haystack_dates": ["2023/04/10 (Mon) 23:07"],
            "haystack_session_ids": ["a", "b"],
            "haystack_sessions": [[{"role": "user", "content": "x"}]],
            "answer_session_ids": [],
        }
    ]
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AdapterError):
        load_longmemeval(path)


def test_lme_unknown_variant_raises():
    with pytest.raises(AdapterError):
        load_longmemeval(variant="xl")


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def test_streaming_reader_matches_json_load():
    path = lme_mod.LONGMEMEVAL_PATHS["oracle"]
    if not path.exists():
        pytest.skip(f"missing {path}")
    streamed = list(lme_mod._stream_json_array(path))
    with path.open(encoding="utf-8") as fh:
        eager = json.load(fh)
    assert len(streamed) == len(eager)
    assert streamed[0] == eager[0]
    assert streamed[-1] == eager[-1]


def test_streaming_reader_handles_edge_shapes(tmp_path: Path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert list(lme_mod._stream_json_array(empty)) == []

    spaced = tmp_path / "spaced.json"
    spaced.write_text('\n\n [ {"a": 1} ,\n {"a": 2} ]\n', encoding="utf-8")
    assert list(lme_mod._stream_json_array(spaced)) == [{"a": 1}, {"a": 2}]

    truncated = tmp_path / "truncated.json"
    truncated.write_text('[{"a": 1}', encoding="utf-8")
    with pytest.raises(AdapterError):
        list(lme_mod._stream_json_array(truncated))

    not_array = tmp_path / "obj.json"
    not_array.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(AdapterError):
        list(lme_mod._stream_json_array(not_array))


def test_streaming_is_lazy(monkeypatch):
    """iter_longmemeval must yield before the file is fully consumed."""
    path = lme_mod.LONGMEMEVAL_PATHS["oracle"]
    if not path.exists():
        pytest.skip(f"missing {path}")
    it = iter_longmemeval()
    first = next(it)
    assert first.source == "longmemeval"
    it.close()


def test_streaming_survives_small_chunks(tmp_path: Path, monkeypatch):
    payload = [{"i": i, "text": "x" * 50} for i in range(20)]
    path = tmp_path / "chunky.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(lme_mod, "_CHUNK", 7)
    assert list(lme_mod._stream_json_array(path)) == payload


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_schema_round_trip_locomo(locomo):
    episode = locomo[0]
    clone = Episode.from_dict(json.loads(json.dumps(episode.to_dict())))
    assert clone.episode_id == episode.episode_id
    assert clone.source == episode.source
    assert clone.messages == episode.messages
    assert clone.qa == episode.qa
    assert clone.meta == episode.meta


def test_schema_round_trip_longmemeval(lme):
    episode = lme[0]
    clone = Episode.from_dict(json.loads(json.dumps(episode.to_dict())))
    assert clone.messages == episode.messages
    assert clone.qa == episode.qa


def test_schema_rejects_non_chronological_messages():
    now = datetime(2023, 5, 8, 13, 56)
    messages = [
        Message("s1", "A", "second", now + timedelta(minutes=1), "D1:2"),
        Message("s1", "B", "first", now, "D1:1"),
    ]
    with pytest.raises(AdapterError):
        Episode("e", messages, [], "locomo")


def test_schema_dataclasses_are_frozen():
    message = Message("s", "A", "t", datetime(2023, 1, 1), "D1:1")
    with pytest.raises(FrozenInstanceError):
        message.text = "mutated"
    item = QAItem("q", "?", "a", "temporal")
    with pytest.raises(FrozenInstanceError):
        item.qid = "other"


def test_assign_timestamps_breaks_identical_session_starts():
    """LongMemEval-s has sessions with byte-identical timestamps. Ordering must
    stay strictly increasing, and the nudge must be reported, not hidden."""
    start = datetime(2023, 5, 21, 9, 54)
    stamps, nudges = assign_timestamps([start, start], [3, 2])
    assert nudges == 1
    flat = [t for session in stamps for t in session]
    assert all(a < b for a, b in zip(flat, flat[1:]))
    assert flat[0] == start


def test_assign_timestamps_compresses_to_fit_the_next_session():
    a = datetime(2023, 5, 1, 12, 0)
    b = a + timedelta(minutes=2)
    stamps, nudges = assign_timestamps([a, b], [10, 1])
    assert nudges == 0
    assert stamps[0][0] == a
    assert stamps[0][-1] < b


def test_shipped_files_need_no_timestamp_nudges(locomo, lme):
    """Both shipped files have wide enough session gaps that every timestamp is
    the real parsed wall clock. If this ever fails, real time is being altered."""
    assert sum(e.meta["timestamp_nudges"] for e in locomo) == 0
    assert sum(e.meta["timestamp_nudges"] for e in lme) == 0


def test_locomo_adversarial_gold_is_a_trap_not_an_abstention_string(locomo):
    """LoCoMo's `adversarial_answer` reads like a real answer: only 2 of 446 are
    abstention-style. Scoring category 5 by string match therefore rewards the
    hallucination the category exists to detect. See REPORT.md section 4."""
    import re

    abstention = re.compile(
        r"no information|not mentioned|cannot|can't|unknown|not enough", re.I
    )
    adversarial = [q for e in locomo for q in e.qa if q.adversarial]
    looks_like_abstention = [q for q in adversarial if abstention.search(q.gold_answer)]
    assert len(adversarial) == 446
    assert len(looks_like_abstention) == 2


def test_locomo_adversarial_questions_misattribute_the_speaker(locomo):
    """333/446 category-5 questions name the speaker who did NOT say the
    evidence, against 31/841 for single_hop. That is the trap mechanism."""

    def misattributed(episode, item):
        speakers = {episode.meta["speaker_a"], episode.meta["speaker_b"]}
        named = {s for s in speakers if s and s.lower() in item.question.lower()}
        evidence = episode.messages_by_ids(item.evidence_ids)
        if not named or not evidence:
            return False
        return not (named & {m.speaker for m in evidence})

    adversarial = sum(
        1 for e in locomo for q in e.qa if q.adversarial and misattributed(e, q)
    )
    single_hop = sum(
        1
        for e in locomo
        for q in e.qa
        if q.category == "single_hop" and misattributed(e, q)
    )
    assert adversarial == 333
    assert single_hop == 31
    assert adversarial / 446 > 10 * (single_hop / 841)


def test_snake_case():
    assert snake_case("single-session-user") == "single_session_user"
    assert snake_case("Temporal Reasoning") == "temporal_reasoning"
    assert snake_case("knowledge-update") == "knowledge_update"
