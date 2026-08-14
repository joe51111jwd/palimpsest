"""What Palimpsest is for, in 60 lines.

    python examples/quickstart.py

No LLM required: the claims are supplied directly so the example runs offline and
deterministically. In real use an extractor produces them from raw messages —
see `examples/with_llm.py`.
"""

from datetime import datetime

from palimpsest import Memory, Message
from palimpsest.types import Claim


def at(day: int) -> datetime:
    return datetime(2023, day // 30 + 1, day % 30 + 1, 12, 0)


CONVERSATION = [
    (1, "I live in New York City.", 0, "city", "New York City"),
    (2, "I work at Globex as a software engineer.", 1, "employer", "Globex"),
    (3, "I'm allergic to penicillin.", 5, "allergic_to", "penicillin"),
    (4, "Just moved to Austin!", 100, "lives_in", "Austin"),          # a synonym
    (5, "My new employer is Pied Piper.", 130, "employer", "Pied Piper"),
]


def main() -> None:
    mem = Memory()

    messages, claims = [], []
    for i, text, day, predicate, value in CONVERSATION:
        msg = Message(session_id="s1", speaker="user", text=text,
                      timestamp=at(day), msg_id=f"m{i}", role="user")
        messages.append(msg)
        claims.append(Claim(entity="user", predicate=predicate, value=value,
                            source_text=text, source_id=msg.msg_id,
                            cardinality="multi" if predicate == "allergic_to" else "single"))
    mem.ingest(messages, claims=claims)

    print("== What is true now ==")
    for fact in mem.facts():
        print("  ", fact.as_line())

    print("\n== Ask where they live ==")
    print(mem.recall("Where do I live?").context)
    print("   ^ 'New York City' is absent. It is not ranked lower — it is not there.")

    print("\n== Ask the same thing about the past ==")
    print(mem.recall("Where do I live?", as_of=at(50)).context)

    print("\n== The palimpsest view: every version, in order ==")
    for fact in mem.timeline("user", "employer"):
        print("  ", fact.as_line())

    print("\n== A fact nothing was hardcoded for ==")
    print(mem.recall("What am I allergic to?").context)

    stats = mem.stats()
    print(f"\n{stats.n_atoms} atoms, {stats.n_current} currently true, "
          f"{stats.n_canonical_predicates} canonical predicates from "
          f"{stats.n_predicates} surface forms")
    print("   ('lives_in' and 'city' collapsed to one attribute, which is the only "
          "reason\n    the move to Austin superseded New York rather than sitting "
          "beside it.)")


if __name__ == "__main__":
    main()
