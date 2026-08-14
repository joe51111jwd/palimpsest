"""Open-world canonicalization of predicates and entities.

This module is the reason the v2 engine exists. An interval ledger only works if
two utterances about the same thing land on the *same key*: if "I moved to
Austin" mints ``lives_in`` and "my city is Austin" mints ``city``, supersession
never fires, both values stay open, and the whole thesis collapses into an
ordinary fact list with extra steps.

A closed predicate whitelist (what v1 did) makes this problem disappear by making
the system useless — it simply refuses to store anything it wasn't hardcoded for.
The real job is to accept *any* predicate an extractor invents and merge the
synonyms online, without a training step and without ever merging two predicates
that mean different things.

Why it is not just "embed the predicate name and threshold the cosine"
---------------------------------------------------------------------
Because that does not work, and we measured it (``bench/canon_probe.py``).
Embedding the bare attribute name ranks *near-misses above true synonyms*:

    lives_in      ~ city                  0.136     <- same thing
    favorite_food ~ least_favorite_food   0.842     <- opposite things
    birth_year    ~ birth_city            0.735     <- different things

Across a trap set, name-only similarity is *inverted* — synonyms average 0.287,
traps average 0.495 — and every one of six trap pairs lands in the **top 3** of
its counterpart's neighbour list. A cosine-threshold canonicalizer on names
confidently merges "favourite food" with "least favourite food".

Embedding the **observed values** alongside the name fixes the ordering, because
synonymous predicates draw values from the same distribution while unrelated ones
do not (``bench/canon_probe2.py``, and see ``profile_text`` below):

    mode          synonyms   traps
    name             0.287   0.495   <- inverted
    values           0.302   0.101   <- correct, 3x gap
    name+values      0.363   0.285   <- correct

It still is not thresholdable — `likes` vs `dislikes` sits at 0.53 on values,
since liking and hating a thing produce identical value distributions. So
similarity is a good *shortlist* signal (the correct cluster is in the top-20 for
100% of predicates) and never a *decision* signal. The decision belongs to the
adjudicator, and the veto belongs to the guards.

Prior art: online canonicalization against a growing predicate vocabulary is not
novel — DIAL-KG (arXiv 2603.20059, DASFAA 2026) does frequency-gated cluster
promotion with LLM schema evaluation and reports only a 0.2–0.9 F1 penalty for
going online. What is contributed here is the interval-ledger consumer of it, and
the measurement that value-profiled similarity is a shortlist-only signal.

Design
------
``canonicalize_predicate(raw, value_type)``:

1. **Alias hit** — exact match against everything seen before. O(1), and it is
   the common case once the vocabulary warms up: novel surface forms become rare
   fast, so steps 2-4 amortize toward zero.
2. **Shortlist** — top-k canonical predicates by embedding rank. Recall-oriented,
   no threshold.
3. **Adjudicate** — an optional callable (in practice one batched LLM call over
   all novel predicates seen in a chunk) picks a candidate or declines. Judgement
   is the LLM's job; it knows ``lives_in`` means ``city`` and knows
   ``favorite_food`` is not ``least_favorite_food``.
4. **Veto** — the deterministic guards below can overrule a merge but never force
   one. They independently catch all six trap pairs, so the engine degrades
   safely to "mint a new predicate" when no adjudicator is configured.
5. **Mint** — otherwise it becomes a new canonical predicate.

Guards (a wrong merge destroys facts, a missed merge only costs a little recall,
so every guard is biased toward *not* merging):

- **Type compatibility.** ``birth_year`` and ``birth_place`` are lexically and
  semantically close but hold a number and a place. Values carry a coarse type;
  incompatible types can never merge.
- **Polarity.** ``likes`` / ``dislikes`` and ``favorite_food`` /
  ``least_favorite_food`` embed almost identically — static embeddings are
  notoriously blind to negation. A lexical negation/extremum check blocks these
  outright, no matter the cosine.
- **Head-noun disagreement.** ``sister_name`` vs ``sister_job`` share a modifier
  but differ in the head; when both sides have a recognizable head token and the
  heads differ and are not themselves synonyms, block.

Everything is online: merging updates a running centroid, no retraining, no
offline clustering pass required. ``rebuild()`` exists for the offline case where
you want order-independence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .embed import Embedder, default_embedder
from .types import ValueType

# Merge thresholds. Predicates are merged more readily than entities: conflating
# two predicates costs recall on one attribute, conflating two *people* corrupts
# every fact about both.
TAU_PREDICATE = 0.86
TAU_ENTITY = 0.94

_NEGATION_MARKERS = (
    "not", "no", "non", "never", "dis", "un", "anti", "ex", "former", "previous",
    "least", "worst", "hate", "hates", "dislike", "dislikes", "avoid", "avoids",
    "allergic", "without", "stopped", "quit",
)
_POSITIVE_MARKERS = (
    "favorite", "favourite", "best", "most", "likes", "like", "loves", "love",
    "prefers", "prefer", "enjoys", "enjoy",
)

# Head nouns that recur across attribute names; used for the head-disagreement
# guard. Grouped so that synonyms do not trip the guard against each other.
_HEAD_SYNONYMS: list[set[str]] = [
    {"name", "called", "named"},
    {"job", "occupation", "profession", "role", "title", "work", "career"},
    {"city", "town", "location", "place", "residence", "home", "lives", "live", "based"},
    {"employer", "company", "firm", "organization", "org", "workplace"},
    {"age", "years", "year", "old"},
    {"birthday", "birthdate", "born", "dob", "date"},
    {"food", "dish", "cuisine", "meal"},
    {"pet", "animal", "dog", "cat"},
    {"car", "vehicle", "auto", "automobile"},
    {"email", "mail", "address"},
    {"phone", "number", "mobile", "cell"},
    {"hobby", "hobbies", "interest", "interests", "pastime"},
    {"partner", "spouse", "husband", "wife", "boyfriend", "girlfriend"},
    {"child", "children", "kid", "kids", "son", "daughter"},
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DATE_RE = re.compile(
    r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})\s*$",
    re.I,
)
_NUMBER_RE = re.compile(r"^\s*-?[\d,]+(\.\d+)?\s*(%|kg|lb|lbs|km|mi|miles|hours?|years?|days?)?\s*$", re.I)
_BOOL_RE = re.compile(r"^\s*(yes|no|true|false)\s*$", re.I)


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def classify_value(value: str) -> ValueType:
    """Coarse, cheap value typing. The LLM extractor may override with a hint."""
    if _BOOL_RE.match(value):
        return "boolean"
    if _DATE_RE.match(value):
        return "date"
    if _NUMBER_RE.match(value):
        return "number"
    return "text"


def types_compatible(a: ValueType, b: ValueType) -> bool:
    """``text`` is the unknown type and unifies with anything; the rest must match."""
    if a == b:
        return True
    return "text" in (a, b)


def _polarity_class(raw: str) -> int:
    """-1 negative, +1 positive, 0 neutral — from the predicate's own wording."""
    toks = set(tokens(raw))
    # Also catch glued prefixes like "dislikes" / "unmarried".
    joined = raw.lower().replace("_", "")
    neg = any(t in _NEGATION_MARKERS for t in toks) or any(
        joined.startswith(p) for p in ("dis", "un", "non", "anti", "ex")
    )
    pos = any(t in _POSITIVE_MARKERS for t in toks)
    if neg and not pos:
        return -1
    if neg and pos:
        # "least_favorite" — negation wins.
        return -1
    return 1 if pos else 0


def _stem(token: str) -> str:
    """Crude singularization so ``year``/``years`` and ``hobby``/``hobbies`` are
    the same head. Deliberately not a real stemmer — over-stemming would merge
    heads that should stay apart."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


#: head stem -> group index, built once from _HEAD_SYNONYMS with plural variants.
_HEAD_GROUP: dict[str, int] = {}
for _i, _group in enumerate(_HEAD_SYNONYMS):
    for _member in _group:
        _HEAD_GROUP.setdefault(_stem(_member), _i)


def _head_token(raw: str) -> str | None:
    """Last content token — English attribute names are head-final
    (``sister_name``, ``favorite_food``, ``home_city``)."""
    toks = tokens(raw)
    return _stem(toks[-1]) if toks else None


def _heads_conflict(a: str, b: str) -> bool:
    ha, hb = _head_token(a), _head_token(b)
    if not ha or not hb or ha == hb:
        return False
    ga, gb = _HEAD_GROUP.get(ha), _HEAD_GROUP.get(hb)
    if ga is not None and gb is not None:
        # Both heads are recognized attribute heads: same group is fine,
        # different groups is a hard conflict (birth_YEAR vs birth_CITY).
        return ga != gb
    # An unrecognized head falls through to the other guards rather than being
    # blocked blindly — the vocabulary is open, so most heads will be unknown.
    return False


#: Prepositional tails mark a ROLE, not the thing itself. `volunteered` is the
#: activity; `volunteered_at` is where it happened. Same stem, different slot.
_ROLE_SUFFIXES = (
    "at", "in", "on", "for", "with", "by", "to", "from", "since", "during",
    "location", "place", "date", "time", "age", "duration", "frequency",
    "count", "number", "reason", "purpose", "start", "end",
)


def _role_suffix_conflict(a: str, b: str) -> bool:
    """True when one predicate is the other plus a role marker.

    Blocks `volunteered` vs `volunteered_at` and `hobby` vs `hobby_duration`
    without needing either head to be in the known-head table.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb or ta == tb:
        return False
    short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
    if long_[: len(short)] != short:
        return False
    tail = long_[len(short) :]
    return bool(tail) and any(t in _ROLE_SUFFIXES for t in tail)


def humanize(raw: str) -> str:
    """``lives_in`` -> ``lives in`` — embeddings are trained on prose, not snake_case."""
    return raw.replace("_", " ").replace("-", " ").strip().lower()


#: How many observed values go into a predicate's profile string.
PROFILE_VALUES = 6


def profile_text(name: str, values: Sequence[str]) -> str:
    """The string a predicate is embedded as.

    Embedding the bare attribute name is the obvious choice and it is measurably
    the wrong one (``bench/canon_probe2.py``). On our trap set, name-only
    similarity is *inverted*: synonym pairs average 0.287 while trap pairs
    average 0.495, because `pet`/`pet_name` and
    `favorite_food`/`least_favorite_food` are lexically near-identical while
    `lives_in`/`city` share no tokens at all.

    Including the observed values flips the ordering, because the values of two
    synonymous predicates are drawn from the same distribution — `lives_in` and
    `city` both hold "Austin", "Boston", "Seattle" — while the values of
    unrelated predicates are not:

        mode          synonyms   traps
        name             0.287   0.495   <- inverted
        values           0.302   0.101   <- correct, 3x gap
        name+values      0.363   0.285   <- correct

    This matches the one rule the canonicalization literature agrees on across
    several independent systems: match on facts, not on the relation name.
    It does NOT make similarity thresholdable — `likes` and `dislikes` still sit
    at 0.53 — which is why the adjudicator and the polarity guard still decide.
    It makes the *shortlist* good, so the adjudicator sees better candidates.
    """
    joined = ", ".join(v for v in values[:PROFILE_VALUES] if v)
    human = name.replace("_", " ").replace("-", " ").strip().lower()
    return f"{human}: {joined}" if joined else human


@dataclass
class CanonicalPredicate:
    cid: int
    name: str
    aliases: set[str] = field(default_factory=set)
    centroid: np.ndarray | None = None
    n_obs: int = 0
    value_types: dict[str, int] = field(default_factory=dict)
    alias_counts: dict[str, int] = field(default_factory=dict)
    values: list[str] = field(default_factory=list)

    @property
    def dominant_type(self) -> ValueType:
        if not self.value_types:
            return "text"
        typed = {k: v for k, v in self.value_types.items() if k != "text"}
        if not typed:
            return "text"
        return max(typed.items(), key=lambda kv: kv[1])[0]  # type: ignore[return-value]

    def observe(
        self,
        raw: str,
        vec: np.ndarray,
        value_type: ValueType,
        value: str = "",
    ) -> None:
        self.aliases.add(raw)
        self.alias_counts[raw] = self.alias_counts.get(raw, 0) + 1
        self.value_types[value_type] = self.value_types.get(value_type, 0) + 1
        if value and value not in self.values:
            self.values.append(value)
        if self.centroid is None:
            self.centroid = vec.astype(np.float32).copy()
        else:
            # Running mean, then renormalize so cosine stays a dot product.
            n = self.n_obs
            self.centroid = (self.centroid * n + vec) / (n + 1)
            norm = float(np.linalg.norm(self.centroid))
            if norm:
                self.centroid = (self.centroid / norm).astype(np.float32)
        self.n_obs += 1
        # The display name is whichever surface form people actually use most.
        self.name = max(self.alias_counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]


@dataclass
class CanonicalEntity:
    cid: int
    name: str
    aliases: set[str] = field(default_factory=set)
    centroid: np.ndarray | None = None
    n_obs: int = 0
    alias_counts: dict[str, int] = field(default_factory=dict)

    def observe(self, raw: str, vec: np.ndarray) -> None:
        self.aliases.add(raw)
        self.alias_counts[raw] = self.alias_counts.get(raw, 0) + 1
        if self.centroid is None:
            self.centroid = vec.astype(np.float32).copy()
        else:
            n = self.n_obs
            self.centroid = (self.centroid * n + vec) / (n + 1)
            norm = float(np.linalg.norm(self.centroid))
            if norm:
                self.centroid = (self.centroid / norm).astype(np.float32)
        self.n_obs += 1
        # Prefer the longest frequent form as the display name ("Maria Santos"
        # over "Maria") — it disambiguates better in a rendered context block.
        self.name = max(self.alias_counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


@dataclass
class MergeDecision:
    canonical_id: int
    minted: bool
    similarity: float = 0.0
    blocked_by: str | None = None
    matched: str | None = None


class PredicateAdjudicator(Protocol):
    """Decides whether a novel predicate means the same as an existing one.

    Implemented in ``palimpsest/extract/adjudicator.py`` by a batched LLM call.
    Returns the chosen candidate's name, or ``None`` to mint a new predicate.
    Guards may veto a merge it proposes; nothing can force one it declines.
    """

    def __call__(
        self, raw: str, candidates: list[str], value_type: ValueType
    ) -> str | None: ...


class Canonicalizer:
    """Online canonicalization over an unbounded predicate/entity vocabulary."""

    #: Entity surface forms that are relational, not identifying. "my sister" is
    #: not a name; it needs to resolve to whoever the sister *is*.
    RELATIONAL = {
        "sister", "brother", "mother", "father", "mom", "dad", "parent", "wife",
        "husband", "partner", "spouse", "son", "daughter", "child", "friend",
        "boss", "manager", "colleague", "coworker", "neighbor", "roommate",
        "girlfriend", "boyfriend", "grandmother", "grandfather", "aunt", "uncle",
        "cousin", "nephew", "niece", "doctor", "therapist", "landlord",
    }

    SELF_ALIASES = {"user", "i", "me", "myself", "you", "self"}

    def __init__(
        self,
        embedder: Embedder | None = None,
        tau_predicate: float = TAU_PREDICATE,
        tau_entity: float = TAU_ENTITY,
        enabled: bool = True,
        adjudicator: PredicateAdjudicator | None = None,
        shortlist_k: int = 20,
    ) -> None:
        self.embedder = embedder or default_embedder()
        self.tau_predicate = tau_predicate
        self.tau_entity = tau_entity
        self.enabled = enabled
        self.adjudicator = adjudicator
        self.shortlist_k = shortlist_k

        self.predicates: list[CanonicalPredicate] = []
        self.entities: list[CanonicalEntity] = []
        self._pred_alias: dict[str, int] = {}
        self._ent_alias: dict[str, int] = {}
        #: relation surface ("sister") -> canonical entity id, learned from
        #: "my sister Maria" style bindings.
        self._relation_binding: dict[str, int] = {}

        self.merge_log: list[tuple[str, str, float, str]] = []

    # ------------------------------------------------------------------ #
    # predicates
    # ------------------------------------------------------------------ #
    def _predicate_vec(self, name: str, values: Sequence[str]) -> np.ndarray:
        return self.embedder.embed_one(profile_text(name, list(values)))

    def shortlist(self, raw: str, k: int = 20, value: str = "") -> list[tuple[int, float]]:
        """Top-k canonical predicate ids by embedding rank. Recall, not decision.

        Measured on a 60-predicate vocabulary: the correct cluster appears here
        for 100% of predicates at k=20. See ``bench/canon_probe.py``.
        """
        if not self.predicates:
            return []
        vec = self._predicate_vec(raw, [value] if value else [])
        return self._ranked(vec, [p.centroid for p in self.predicates])[:k]

    def canonicalize_predicate(
        self,
        raw: str,
        value_type: ValueType = "text",
        *,
        value: str = "",
        adjudicate: bool = True,
    ) -> MergeDecision:
        raw = raw.strip().lower().replace(" ", "_")
        if not raw:
            raw = "attribute"

        hit = self._pred_alias.get(raw)
        if hit is not None:
            cand = self.predicates[hit]
            vec = self._predicate_vec(cand.name, [*cand.values, value])
            cand.observe(raw, vec, value_type, value)
            return MergeDecision(canonical_id=hit, minted=False, similarity=1.0, matched=raw)

        # A brand-new surface form is profiled with the single value we have.
        vec = self._predicate_vec(raw, [value] if value else [])

        if not self.enabled or not self.predicates:
            return self._mint_predicate(raw, vec, value_type, value)

        order = self._ranked(vec, [p.centroid for p in self.predicates])

        # -- 1. adjudicated merge (the accurate path) --------------------- #
        if adjudicate and self.adjudicator is not None:
            candidates = [self.predicates[i].name for i, _ in order[: self.shortlist_k]]
            try:
                choice = self.adjudicator(raw, candidates, value_type)
            except Exception:
                choice = None
            if choice:
                cid = self._pred_alias.get(choice.strip().lower().replace(" ", "_"))
                if cid is not None:
                    cand = self.predicates[cid]
                    blocked = self._predicate_block_reason(raw, value_type, cand)
                    if blocked:
                        # The guards can overrule the model, never the reverse.
                        self.merge_log.append((raw, cand.name, 0.0, f"veto:{blocked}"))
                    else:
                        sim = next((s for i, s in order if i == cid), 0.0)
                        cand.observe(raw, vec, value_type, value)
                        self._pred_alias[raw] = cand.cid
                        self.merge_log.append((raw, cand.name, sim, "adjudicated"))
                        return MergeDecision(
                            canonical_id=cand.cid, minted=False,
                            similarity=sim, matched=cand.name,
                        )

        # -- 2. conservative threshold fallback (no adjudicator available) - #
        # Deliberately strict: with static embeddings a permissive threshold
        # merges antonyms, so when in doubt this mints instead.
        for idx, sim in order:
            if sim < self.tau_predicate:
                break
            cand = self.predicates[idx]
            blocked = self._predicate_block_reason(raw, value_type, cand)
            if blocked:
                self.merge_log.append((raw, cand.name, sim, f"blocked:{blocked}"))
                continue
            cand.observe(raw, vec, value_type, value)
            self._pred_alias[raw] = cand.cid
            self.merge_log.append((raw, cand.name, sim, "merged"))
            return MergeDecision(
                canonical_id=cand.cid, minted=False, similarity=sim, matched=cand.name
            )

        best_sim = order[0][1] if order else 0.0
        decision = self._mint_predicate(raw, vec, value_type, value)
        decision.similarity = best_sim
        return decision

    def _predicate_block_reason(
        self, raw: str, value_type: ValueType, cand: CanonicalPredicate
    ) -> str | None:
        if not types_compatible(value_type, cand.dominant_type):
            return "value_type"
        # `text` normally means "unknown type" and unifies with anything. But once
        # a predicate has several observations that are ALL free text, its
        # text-ness is evidence rather than ignorance, and a numeric or dated
        # incoming value is a genuine mismatch. This is what separates
        # `instrument_played` ("violin") from `years_playing_instrument` ("20").
        if value_type != "text" and cand.dominant_type == "text":
            observed = cand.value_types.get("text", 0)
            if observed >= 2 and cand.n_obs == observed:
                return "value_type_evidenced"
        raw_pol = _polarity_class(raw)
        for alias in cand.aliases:
            if _polarity_class(alias) != raw_pol:
                return "polarity"
            if _heads_conflict(raw, alias):
                return "head_noun"
            if _role_suffix_conflict(raw, alias):
                return "role_suffix"
        return None

    def _mint_predicate(
        self, raw: str, vec: np.ndarray, value_type: ValueType, value: str = ""
    ) -> MergeDecision:
        cp = CanonicalPredicate(cid=len(self.predicates), name=raw)
        cp.observe(raw, vec, value_type, value)
        self.predicates.append(cp)
        self._pred_alias[raw] = cp.cid
        return MergeDecision(canonical_id=cp.cid, minted=True)

    # ------------------------------------------------------------------ #
    # entities
    # ------------------------------------------------------------------ #
    def canonicalize_entity(self, raw: str) -> MergeDecision:
        surface = raw.strip()
        low = surface.lower()
        if not low:
            low, surface = "user", "user"

        # First person always collapses to one entity.
        if low in self.SELF_ALIASES or low.startswith("my "):
            stripped = low[3:].strip() if low.startswith("my ") else low
            if not stripped or stripped in self.SELF_ALIASES:
                return self._resolve_alias_or_mint("user", "user")
            # "my sister" -> whoever sister is bound to, else a relation entity.
            bound = self._relation_binding.get(stripped)
            if bound is not None:
                self.entities[bound].observe(stripped, self.embedder.embed_one(stripped))
                return MergeDecision(canonical_id=bound, minted=False, matched=stripped)
            low, surface = stripped, stripped

        hit = self._ent_alias.get(low)
        if hit is not None:
            self.entities[hit].observe(low, self.embedder.embed_one(low))
            return MergeDecision(canonical_id=hit, minted=False, similarity=1.0, matched=low)

        # Name-containment: "Maria" and "Maria Santos" are the same person when
        # one is a token-subset of the other and both look like names.
        contained = self._name_containment(low)
        if contained is not None:
            self.entities[contained].observe(low, self.embedder.embed_one(low))
            self._ent_alias[low] = contained
            return MergeDecision(canonical_id=contained, minted=False, similarity=1.0)

        vec = self.embedder.embed_one(low)
        if self.enabled and self.entities:
            order = self._ranked(vec, [e.centroid for e in self.entities])
            if order and order[0][1] >= self.tau_entity:
                idx, sim = order[0]
                # Never merge a relational word into a name purely on cosine.
                if low not in self.RELATIONAL and not any(
                    a in self.RELATIONAL for a in self.entities[idx].aliases
                ):
                    self.entities[idx].observe(low, vec)
                    self._ent_alias[low] = self.entities[idx].cid
                    return MergeDecision(
                        canonical_id=self.entities[idx].cid, minted=False, similarity=sim
                    )
        return self._mint_entity(low, vec)

    def _resolve_alias_or_mint(self, low: str, surface: str) -> MergeDecision:
        hit = self._ent_alias.get(low)
        if hit is not None:
            return MergeDecision(canonical_id=hit, minted=False, similarity=1.0)
        return self._mint_entity(low, self.embedder.embed_one(low))

    def _name_containment(self, low: str) -> int | None:
        toks = set(tokens(low))
        if not toks or low in self.RELATIONAL:
            return None
        best: tuple[int, int] | None = None
        for ent in self.entities:
            for alias in ent.aliases:
                if alias in self.RELATIONAL:
                    continue
                atoks = set(tokens(alias))
                if not atoks:
                    continue
                if toks == atoks:
                    continue
                if toks < atoks or atoks < toks:
                    overlap = len(toks & atoks)
                    if best is None or overlap > best[1]:
                        best = (ent.cid, overlap)
        return best[0] if best else None

    def _mint_entity(self, low: str, vec: np.ndarray) -> MergeDecision:
        ce = CanonicalEntity(cid=len(self.entities), name=low)
        ce.observe(low, vec)
        self.entities.append(ce)
        self._ent_alias[low] = ce.cid
        return MergeDecision(canonical_id=ce.cid, minted=True)

    def bind_relation(self, relation: str, entity_raw: str) -> None:
        """Learn "my sister Maria": bind the relation word to a named entity.

        After this, "my sister" and "Maria" resolve to one canonical entity, which
        is what makes a 2-hop question ("what does my sister do?") a 1-hop lookup.
        """
        relation = relation.strip().lower()
        if not relation:
            return
        decision = self.canonicalize_entity(entity_raw)
        self._relation_binding[relation] = decision.canonical_id
        self._ent_alias[relation] = decision.canonical_id
        self.entities[decision.canonical_id].aliases.add(relation)

    # ------------------------------------------------------------------ #
    # lookup helpers (used by retrieval)
    # ------------------------------------------------------------------ #
    def predicate_matrix(self) -> np.ndarray:
        if not self.predicates:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        return np.stack(
            [
                p.centroid
                if p.centroid is not None
                else np.zeros(self.embedder.dim, dtype=np.float32)
                for p in self.predicates
            ]
        ).astype(np.float32)

    def entity_matrix(self) -> np.ndarray:
        if not self.entities:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        return np.stack(
            [
                e.centroid
                if e.centroid is not None
                else np.zeros(self.embedder.dim, dtype=np.float32)
                for e in self.entities
            ]
        ).astype(np.float32)

    def lookup_predicate(self, raw: str) -> int | None:
        return self._pred_alias.get(raw.strip().lower().replace(" ", "_"))

    def lookup_entity(self, raw: str) -> int | None:
        return self._ent_alias.get(raw.strip().lower())

    @staticmethod
    def _ranked(vec: np.ndarray, centroids: list[np.ndarray | None]) -> list[tuple[int, float]]:
        if not centroids:
            return []
        mat = np.stack(
            [c if c is not None else np.zeros_like(vec) for c in centroids]
        ).astype(np.float32)
        sims = mat @ vec.astype(np.float32)
        order = np.argsort(sims)[::-1]
        return [(int(i), float(sims[i])) for i in order]

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        merged = sum(1 for *_, kind in self.merge_log if kind == "merged")
        blocked = sum(1 for *_, kind in self.merge_log if kind.startswith("blocked"))
        reasons: dict[str, int] = {}
        for *_, kind in self.merge_log:
            if kind.startswith("blocked:"):
                reasons[kind.split(":", 1)[1]] = reasons.get(kind.split(":", 1)[1], 0) + 1
        return {
            "canonical_predicates": len(self.predicates),
            "predicate_surface_forms": len(self._pred_alias),
            "canonical_entities": len(self.entities),
            "entity_surface_forms": len(self._ent_alias),
            "merges": merged,
            "blocked": blocked,
            "blocked_reasons": reasons,
            "compression": (
                len(self._pred_alias) / len(self.predicates) if self.predicates else 1.0
            ),
        }
