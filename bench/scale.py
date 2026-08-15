"""Scaling harness — does the engine survive a store it was never benchmarked on?

Benchmark episodes are tiny (LoCoMo ~600 messages, LongMemEval-S ~1.5k). A memory
system that only holds a benchmark episode is a demo, so this synthesises stores
of 1k / 10k / 100k messages and reports ingest throughput, query latency
percentiles and resident memory at each size.

Nothing here is dataset-specific: the text is generated from a Zipfian vocabulary
with named entities and attribute claims mixed in, which is the shape of a real
chat log, not the shape of any benchmark.

    python -m bench.scale --sizes 1000,10000 --queries 100
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import statistics
import sys
import time
from datetime import datetime, timedelta

from palimpsest.store import Memory
from palimpsest.types import Claim, Message

# ---------------------------------------------------------------------- #
# synthetic corpus
# ---------------------------------------------------------------------- #

_FIRST = """maria james sofia liam noah olivia emma ava william henry lucas mila
ella jack owen leo aria luna chloe caleb ruby ethan grace hazel isaac jonah kayla
logan mason nora oscar piper quinn rosa simon tessa uma victor wendy xavier yara
zane amara bruno cora dario elena fabio gina hugo iris""".split()

_LAST = """santos rivera chen okafor patel novak silva haddad kim moretti dubois
ferreira ivanov nakamura oconnor petrov rahman schmidt tanaka vargas walsh""".split()

_PREDICATES = [
    ("city", "moved to", ["austin", "boston", "denver", "seattle", "lisbon", "osaka"]),
    ("employer", "started working at", ["globex", "initech", "umbrella", "hooli", "acme"]),
    ("job_title", "got promoted to", ["analyst", "designer", "engineer", "manager"]),
    ("pet", "adopted", ["a tabby cat", "a beagle", "a corn snake", "two goldfish"]),
    ("hobby", "took up", ["pottery", "bouldering", "kendo", "birdwatching", "baking"]),
    ("favorite_food", "loves", ["ramen", "khachapuri", "tamales", "pho", "gnocchi"]),
    ("car", "bought", ["a used volvo", "an electric hatchback", "a cargo bike"]),
    ("phone", "switched to", ["a pixel", "an old iphone", "a fairphone"]),
    ("gym", "joined", ["riverside fitness", "the climbing barn", "yoga collective"]),
    ("degree", "is studying", ["marine biology", "linguistics", "civil engineering"]),
]

_TOPIC_WORDS = """
project deadline commute rehearsal invoice landlord thermostat sourdough transit
kettle allergy migraine passport visa harvest scaffold sculpture archive telescope
festival marathon canoe hammock ledger receipt parcel courier gasket radiator
notebook printer podcast rehearse subscription warranty firmware plumber cinema
espresso bicycle balcony rooftop compost knitting darkroom vinyl trombone glacier
kayak orchard vineyard bakery pharmacy dentist optician tailor cobbler florist
antenna generator sandbag driveway porch pantry attic basement gutter shingle
proposal budget quarter roadmap retro standup incident postmortem migration
""".split()

_FILLER = """
honestly i think we should probably wait until the weekend before deciding
anything because the whole thing has been dragging on and nobody wants to redo it
again after last time when everything fell apart halfway through and we had to
start over from almost nothing which was frustrating for everyone involved
""".split()


def _zipf_vocab(rng: random.Random, size: int = 4000) -> list[str]:
    base = _TOPIC_WORDS + _FILLER
    vocab = list(base)
    while len(vocab) < size:
        vocab.append("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 9))))
    return vocab


def synth(n_messages: int, seed: int = 7, prefix: str = "m") -> tuple[list[Message], list[Claim]]:
    """A chat log of ``n_messages`` turns with entities, attributes and drift."""
    rng = random.Random(seed)
    vocab = _zipf_vocab(rng)
    weights = [1.0 / (i + 1) for i in range(len(vocab))]
    cum: list[float] = []
    total = 0.0
    for w in weights:
        total += w
        cum.append(total)

    def word() -> str:
        import bisect

        return vocab[bisect.bisect(cum, rng.random() * total)]

    # a cast that grows with the corpus, as a real long-lived store's would
    n_people = max(8, int(n_messages ** 0.55))
    people = [f"{rng.choice(_FIRST)} {rng.choice(_LAST)}" for _ in range(n_people)]

    t0 = datetime(2019, 1, 1, 9, 0)
    messages: list[Message] = []
    claims: list[Claim] = []
    for i in range(n_messages):
        ts = t0 + timedelta(minutes=11 * i)
        role = "user" if i % 2 == 0 else "assistant"
        mid = f"{prefix}{i}"
        person = rng.choice(people)
        body = " ".join(word() for _ in range(rng.randint(12, 34)))
        if rng.random() < 0.28:
            pred, verb, values = rng.choice(_PREDICATES)
            value = rng.choice(values)
            subject = "i" if rng.random() < 0.5 else person
            text = f"{subject} {verb} {value}. {body}"
            claims.append(
                Claim(
                    entity="user" if subject == "i" else person,
                    predicate=pred,
                    value=value,
                    source_text=text[:120],
                    source_id=mid,
                    confidence=0.9,
                )
            )
        else:
            text = f"{person} mentioned {word()} — {body}"
        messages.append(
            Message(
                session_id=f"s{i // 50}",
                speaker=role,
                text=text,
                timestamp=ts,
                msg_id=mid,
                role=role,
            )
        )
    return messages, claims


def synth_queries(messages: list[Message], n: int, seed: int = 11) -> list[tuple[str, datetime]]:
    """Questions built from terms that really occur, asked at real timestamps."""
    rng = random.Random(seed)
    out: list[tuple[str, datetime]] = []
    stems = [
        "where do i live",
        "who do i work for",
        "what is my job title",
        "what pet do i have",
        "what hobby did i take up",
        "what is my favorite food",
        "what car did i buy",
    ]
    for i in range(n):
        msg = rng.choice(messages)
        toks = [t for t in msg.text.split() if len(t) > 4][:4]
        if i % 3 == 0:
            q = rng.choice(stems)
        else:
            q = "what did we say about " + " ".join(toks)
        out.append((q, msg.timestamp))
    return out


# ---------------------------------------------------------------------- #
# measurement
# ---------------------------------------------------------------------- #


def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def run_size(n: int, n_queries: int, batch: int = 50) -> dict:
    gc.collect()
    rss_before = rss_mb()
    messages, claims = synth(n)
    by_src: dict[str, list[Claim]] = {}
    for c in claims:
        by_src.setdefault(c.source_id, []).append(c)

    mem = Memory()

    # CPU time, not wall clock: this machine is shared with other jobs and wall
    # clock measures their load as much as ours. The workload is single-threaded
    # apart from the embedder, so CPU time is the honest comparable number.
    t0 = time.process_time()
    for i in range(0, len(messages), batch):
        chunk = messages[i : i + batch]
        chunk_claims = [c for m in chunk for c in by_src.get(m.msg_id, ())]
        mem.ingest(chunk, claims=chunk_claims)
    mem.index.build()
    ingest_s = time.process_time() - t0

    queries = synth_queries(messages, n_queries)
    # warm the embedder cache for the query strings the same way every run does
    mem.recall(queries[0][0], as_of=queries[0][1], known_at=queries[0][1])

    lat: list[float] = []
    for q, ts in queries:
        t = time.process_time()
        mem.recall(q, as_of=ts, known_at=ts)
        lat.append((time.process_time() - t) * 1000)
    lat.sort()

    # incremental round: one more session + one query at full store size
    inc: list[float] = []
    extra, extra_claims = synth(batch * 10, seed=99, prefix="x")
    extra_by_src: dict[str, list[Claim]] = {}
    for c in extra_claims:
        extra_by_src.setdefault(c.source_id, []).append(c)
    for j in range(0, len(extra), batch):
        t = time.process_time()
        chunk = extra[j : j + batch]
        mem.ingest(chunk, claims=[c for m in chunk for c in extra_by_src.get(m.msg_id, ())])
        mem.recall(queries[j % len(queries)][0])
        inc.append((time.process_time() - t) * 1000)

    gc.collect()
    stats = mem.stats()
    out = {
        "messages": n,
        "claims": len(claims),
        "atoms": stats.n_atoms,
        "entities": stats.n_entities,
        "predicates": stats.n_canonical_predicates,
        "ingest_s": round(ingest_s, 3),
        "ingest_msg_per_s": round(n / ingest_s, 1),
        "q_p50_ms": round(lat[len(lat) // 2], 2),
        "q_p95_ms": round(lat[int(len(lat) * 0.95)], 2),
        "q_max_ms": round(lat[-1], 2),
        "incr_round_p50_ms": round(statistics.median(inc), 2),
        "rss_mb": round(rss_mb(), 1),
        "rss_delta_mb": round(rss_mb() - rss_before, 1),
        "index_mb": round(stats.index_bytes / 1e6, 2),
    }
    del mem
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,10000,100000")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    hdr = (
        f"{'msgs':>7} {'ingest_s':>9} {'msg/s':>8} {'p50_ms':>8} {'p95_ms':>8} "
        f"{'max_ms':>8} {'incr_ms':>8} {'rss_mb':>8} {'idx_mb':>7} {'atoms':>7} {'ents':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in [int(x) for x in args.sizes.split(",") if x]:
        r = run_size(s, args.queries, args.batch)
        rows.append(r)
        print(
            f"{r['messages']:>7} {r['ingest_s']:>9.2f} {r['ingest_msg_per_s']:>8.1f} "
            f"{r['q_p50_ms']:>8.2f} {r['q_p95_ms']:>8.2f} {r['q_max_ms']:>8.2f} "
            f"{r['incr_round_p50_ms']:>8.2f} {r['rss_mb']:>8.1f} {r['index_mb']:>7.2f} "
            f"{r['atoms']:>7} {r['entities']:>6}",
            flush=True,
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
