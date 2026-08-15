# Palimpsest — site design system

Everything on this site is built from `assets/site.css` and `assets/site.js`.
Plain static HTML. No build step, no framework, no npm, no external requests of any kind.
Every page must open correctly by double-clicking the `.html` file **and** deploy to Vercel
as static output.

`index.html` is the reference implementation. When something here is ambiguous, read it there.

---

## 0. Rules for page authors

These are not stylistic preferences. Breaking one breaks the site.

1. **Do not edit `assets/site.css` or `assets/site.js`.** Build your page against the classes
   below. If you need something that does not exist, report it — do not add a `<style>` block
   and do not add a second stylesheet.
2. **No inline `style=` attributes.** Spacing utilities (`.mt-3`, `.mt-5`, `.stack`, …) cover it.
3. **No external requests.** No font CDNs, no analytics, no remote images, no CDN scripts.
   System font stacks only. The site must work offline.
4. **One `<h1>` per page**, in the hero. Section headings are `<h2>`.
5. **Every number must match the project's stated facts exactly.** Do not round differently, do
   not drop a confidence interval, do not omit a loss. If a number is not in the source facts,
   do not invent it — leave `<span class="todo">TODO: …</span>`.
6. **The caveat travels with the headline.** Wherever a LongMemEval headline number appears, the
   `.callout--caveat` stating that the interval overlaps the runner-up must appear on the same
   page, near it. Same for LoCoMo: we lose it, and the page says so.
7. **Voice**: plain, technical, unhurried. A skeptical engineer is the reader. No marketing
   language, no emoji as decoration, no "we" implying a company — this is one person's project.
8. **Tables always go inside `.table-wrap`.** The page body must never scroll horizontally.
9. **Check both themes and a 375px-wide viewport** before you call a page done.

---

## 1. The idea

A palimpsest is a manuscript that was scraped and rewritten, where the earlier text stays
legible underneath. The way palimpsests are actually read today is **multispectral imaging**:
the sheet is lit at wavelengths the naked eye cannot use, and the erased undertext is recovered
as a glow against a cool grey plate.

That gives the site its split of roles, which is the whole design concept:

- **The site is the imaging apparatus.** Cool, instrumental, quiet. Plate-grey ground, hairline
  rules, monospaced labels and dates, numbers in tabular figures, nothing centred, nothing
  decorated.
- **The data is the manuscript.** Current values sit on top at full ink. Superseded values sit
  beneath them, stepped in, washed with an erasure hatch, glowing in recovered-text amber.

This is also literally what the engine does — a new value closes the previous interval instead
of coexisting with it, and nothing is ever deleted, only closed — so the metaphor is load-bearing
rather than ornamental. It is expressed as a real, reusable component: **`.stratum`** (§6).

Explicitly avoided: warm cream + serif + terracotta, purple-to-blue gradient heroes, Inter and
Space Grotesk, emoji section markers, rounded cards with an accent rail, centred-by-default
layout.

---

## 2. Palette

The complete light palette is defined on bare `:root`. Dark mode **only redefines** those same
token names, in two places that must stay identical: `@media (prefers-color-scheme: dark)
:root:not([data-theme="light"])` and `:root[data-theme="dark"]`. No colour is ever declared for
the first time inside a media query, and `body` has an explicit background.

### Ground — cool plate grey, blue-biased

Never white, never cream. The page is a photographic plate.

| token | light | dark | use |
|---|---|---|---|
| `--c-bg` | `#e9edf2` | `#0b1016` | page ground |
| `--c-bg-sunken` | `#dfe5ec` | `#070b10` | footer, sunken sections |
| `--c-surface` | `#f4f7fa` | `#121a23` | panels, tables, cards |
| `--c-surface-2` | `#fbfcfe` | `#18222d` | raised: the current stratum layer, hover |
| `--c-surface-inset` | `#e3e9f0` | `#0e151d` | code wells, panel and table headers |

### Ink — iron gall, oxidised toward blue-black

| token | light | dark | use |
|---|---|---|---|
| `--c-ink` | `#101821` | `#dce4ec` | body text, current values |
| `--c-ink-2` | `#33404f` | `#adbac7` | secondary prose, ledes |
| `--c-ink-3` | `#4e5c6a` | `#8695a4` | labels, captions, table headers |
| `--c-ink-4` | `#5a6775` | `#7d8b99` | faintest text still meant to be read |
| `--c-ink-inverse` | `#f4f7fa` | `#0b1016` | text on the filled button |

### Lines

| token | light | dark | use |
|---|---|---|---|
| `--c-border` | `#c8d2dd` | `#26313d` | standard hairline |
| `--c-border-soft` | `#d9e1e9` | `#1c2530` | inner rules, row separators |
| `--c-border-strong` | `#9fadbc` | `#3b4959` | card top rule, rail rule (decorative) |
| `--c-control-border` | `#6f7e8d` | `#6b7a89` | **the boundary of an actual control** |

`--c-control-border` exists so buttons read as buttons without darkening every hairline in the
system. Use it only on interactive borders.

### Accent — recovered undertext. The only accent.

Amber means *recovered from beneath*. It is used for superseded values, the current-layer tick,
the mandatory caveat, focus rings, and the marked row in a results table. Nothing else.

| token | light | dark | use |
|---|---|---|---|
| `--c-accent` | `#b8720c` | `#f0b04a` | ticks, rules, marks — not text on light |
| `--c-accent-text` | `#8a5300` | `#f0b04a` | amber **text** (AA-safe in both themes) |
| `--c-accent-strong` | `#6d4200` | `#f7c877` | hover/emphasis |
| `--c-accent-soft` | `#f2e6d1` | `#2a2113` | the caveat callout's fill, superseded pills |
| `--c-accent-line` | `#d9b479` | `#6d5323` | link underlines, amber hairlines |
| `--c-focus` | `#8a5300` | `#f0b04a` | `:focus-visible` outline |

### Signed deltas — reserved

`--c-pos` (`#0f5c4a` / `#5cc3a8`) and `--c-neg` (`#8c2f28` / `#e08a83`) are used **only** for
signed numbers in result tables and the "where it loses" callout border. They are never a brand
colour and never decorate anything.

### Washes

`--tint-erasure` is the scraped-page hatch laid over superseded strata and the hero's plate scan
lines. `--tint-glow` is the amber wash behind our own row in a results table.

### Contrast (measured, WCAG 2.1)

Every text token clears AA (4.5:1) against every surface it is allowed to sit on, in both themes.

| pair | light | dark |
|---|---:|---:|
| `--c-ink` on `--c-bg` | 15.2 | 14.9 |
| `--c-ink-2` on `--c-bg` | 9.0 | 9.7 |
| `--c-ink-3` on `--c-surface-inset` (worst case) | 4.8 | 6.0 |
| `--c-ink-4` on `--c-surface-inset` (worst case) | 4.7 | 4.6 |
| `--c-accent-text` on `--c-bg` | 5.4 | 10.0 |
| `--c-accent-text` on `--c-accent-soft` | 5.1 | 8.3 |
| `--c-ink-inverse` on filled `.btn` | 16.6 | 14.9 |

---

## 3. Type

**System stacks only.** Nothing is loaded over the network.

- `--font-sans` — `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
  "Noto Sans", Arial, sans-serif`. Prose and headings.
- `--font-mono` — `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Roboto Mono",
  "Liberation Mono", monospace`. The **apparatus**: eyebrows, section numbers, plate labels,
  dates and intervals, table headers, numeric cells, pills, buttons, code.

The pairing is the concept, not a font choice: everything the instrument says is monospaced;
everything the human says is prose. That is why headings are set in the sans at tight tracking
(`--tr-tight`, `-0.021em`) and never in a serif — a serif here would drag the page toward the
generic "manuscript" look this project is avoiding, and the manuscript is the *data*, not the UI.

Scale (`--fs-*`): `2xs .6875rem` · `xs .75` · `sm .8125` · `ui .875` · `base 1` · `md 1.0625`
(body) · `lg` · `xl` · `2xl` · `3xl` · `4xl`. The last five are `clamp()`-fluid.
Line heights: `--lh-tight 1.15` (h1), `--lh-snug 1.3` (headings), `--lh-body 1.62` (prose).

Numbers are **always** `font-variant-numeric: tabular-nums` in mono — `.num` and `.mono` do this,
and `table.data` does it for the whole table.

---

## 4. Layout

- `.wrap` — the page container: `max-width: 76rem`, fluid gutter `clamp(1rem, 4vw, 2.5rem)`.
- **Nothing is centred by default.** No `text-align: center` exists in the stylesheet.
- **The apparatus rail.** Every content section is a `.section-grid`: a narrow left column
  (`.section-rail`, 11rem) carrying a two-digit section number and a short label in mono, and the
  content to its right. Below 62rem the rail stacks above the content; above it, the rail is
  sticky. This is the site's signature and it is why the eye always knows where it is.
- Section numbers restart at `01` on each page and run in reading order.
- Prose measure is `--max-prose` (68ch) via `.prose`. Tables, panels and code may run full width.
- Spacing scale `--sp-1 … --sp-10` (0.25rem → 7rem). Stacks: `.stack`, `.stack-sm`, `.stack-lg`.
- Grids: `.grid-2`, `.grid-3` (1 → 2 → 3 columns), `.compare` (1 → 2 equal columns at 56rem).
- Radii are deliberately small: `--r-1 2px`, `--r-2 3px`, `--r-3 5px`.

---

## 5. Page boilerplate

Copy this for every new page. Change the `<title>`, the `<meta name="description">`, the
`aria-current="page"` on the nav item, and the `<main>` contents.

`site.js` is loaded in `<head>` **without `defer`** on purpose: it applies the stored theme
during head parsing so there is no flash of the wrong theme. Everything else it does waits for
`DOMContentLoaded`.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How it works — Palimpsest</title>
<meta name="description" content="One sentence, plain, no marketing.">
<link rel="stylesheet" href="assets/site.css">
<script src="assets/site.js"></script>
</head>
<body>

<a class="skip-link" href="#main">Skip to content</a>

<header class="site-header" data-nav="closed">
  <div class="wrap site-header__inner">
    <a class="brand" href="index.html">
      <span class="brand__name">Palimpsest</span>
      <span class="brand__note">bitemporal claim-interval memory</span>
    </a>
    <nav class="nav" aria-label="Primary">
      <ul class="nav-list">
        <li><a href="how-it-works.html" aria-current="page">How it works</a></li>
        <li><a href="benchmarks.html">Benchmarks</a></li>
        <li><a href="field.html">The field</a></li>
        <li><a href="audit.html">Audit</a></li>
        <li><a href="docs.html">Docs</a></li>
        <li><a href="https://github.com/joe51111jwd/palimpsest">GitHub</a></li>
      </ul>
    </nav>
    <button class="nav-toggle" type="button" data-nav-toggle aria-expanded="false">Menu</button>
    <button class="theme-toggle" type="button" data-theme-toggle>Dark</button>
  </div>
</header>

<main id="main">

  <section class="hero">
    <div class="wrap">
      <p class="eyebrow">Mechanism</p>
      <h1 class="hero__title">One h1 per page. State the point, do not tease it.</h1>
      <p class="hero__lede">One or two sentences of orientation.</p>
    </div>
  </section>

  <section class="section" id="something">
    <div class="wrap section-grid">
      <div class="section-rail"><b>01</b><span>Short label</span></div>
      <div class="stack-lg">
        <div class="prose">
          <h2>Section heading.</h2>
          <p>Body copy.</p>
        </div>
      </div>
    </div>
  </section>

</main>

<footer class="site-footer">
  <div class="wrap site-footer__grid">
    <div>
      <h2>Palimpsest</h2>
      <p class="measure-short">An open-world bitemporal claim-interval memory engine for AI
        agents. A palimpsest is a manuscript scraped and rewritten, where the earlier text stays
        legible underneath. So is this.</p>
      <p class="meta mt-3">Apache-2.0 &middot; alpha &middot; the API may change</p>
    </div>
    <div>
      <h2>Read</h2>
      <ul>
        <li><a href="how-it-works.html">How it works</a></li>
        <li><a href="benchmarks.html">Benchmarks</a></li>
        <li><a href="field.html">The field</a></li>
        <li><a href="audit.html">Audit</a></li>
      </ul>
    </div>
    <div>
      <h2>Use</h2>
      <ul>
        <li><a href="docs.html">Docs &amp; API</a></li>
        <li><a href="https://github.com/joe51111jwd/palimpsest">GitHub</a></li>
        <li><a href="https://pypi.org/project/palimpsest-memory/">PyPI</a></li>
        <li><a href="about.html">About</a></li>
      </ul>
    </div>
  </div>
</footer>

</body>
</html>
```

---

## 6. Component reference

Copy-pasteable. Every class below already exists in `assets/site.css`.

### 6.1 `.stratum` — the central component

A stacked block: the current value on top at full ink with an amber tick, superseded values
beneath it, stepped in, hatched with the erasure wash, their values glowing in recovered-text
amber. Use it anywhere a version chain is shown — it is the project's visual argument and should
appear on more than one page.

**DOM order is current-first, then superseded newest-first.** Depth (the progressive inset and
fade) is handled by sibling chaining and reads clearly for up to four superseded layers.

```html
<div class="stratum">
  <div class="stratum__key">
    <b>user</b> <span>&middot;</span> <b>employer</b>
    <span>&middot; version chain, valid time</span>
  </div>
  <div class="stratum__layers">
    <div class="stratum-layer is-current">
      <span class="stratum-layer__value">Cyberdyne</span>
      <span class="pill pill--current">current</span>
      <span class="stratum-layer__range">2024-01-08 &rarr; open</span>
    </div>
    <div class="stratum-layer is-superseded">
      <span class="stratum-layer__value">Globex</span>
      <span class="pill pill--superseded">superseded</span>
      <span class="stratum-layer__range">2022-08-15 &rarr; 2024-01-08</span>
    </div>
    <div class="stratum-layer is-superseded">
      <span class="stratum-layer__value">Initech</span>
      <span class="pill pill--superseded">superseded</span>
      <span class="stratum-layer__range">2021-03-02 &rarr; 2022-08-15</span>
    </div>
  </div>
  <div class="stratum__note">Optional footnote inside the block.</div>
</div>
```

**Corrections vs. changes.** A transaction-time close — *"I was never at Globex, I misspoke"* —
is not a supersession. Use `.is-corrected`, which strikes the value through, and
`.pill--corrected`:

```html
<div class="stratum-layer is-superseded is-corrected">
  <span class="stratum-layer__value">Globex</span>
  <span class="pill pill--corrected">corrected</span>
  <span class="stratum-layer__range">never true</span>
</div>
```

| class | meaning |
|---|---|
| `.stratum` | the block |
| `.stratum__key` | header: the `(entity, predicate)` key and which time axis is shown |
| `.stratum__layers` | wrapper for the layers (required — depth chaining relies on it) |
| `.stratum-layer` | one version |
| `.is-current` | full ink, raised surface, amber tick |
| `.is-superseded` | erasure hatch, amber value, stepped inset |
| `.is-corrected` | struck through — never true, not merely old |
| `.stratum-layer__value` / `__range` | the value; the interval (mono, tabular) |
| `.stratum__note` | optional footer inside the block |

### 6.2 Panels and the side-by-side

```html
<div class="compare">
  <div class="panel">
    <div class="panel__head">
      <span class="panel__title">Vector store &rarr; context</span>
      <span class="pill">3 chunks</span>
    </div>
    <div class="panel__body">
      <div class="chunk">
        <span class="chunk__text">&ldquo;I work at Globex now.&rdquo;</span>
        <span class="chunk__score">match</span>
      </div>
    </div>
    <div class="panel__foot">What the reader should conclude.</div>
  </div>

  <div class="panel">
    <div class="panel__head"><span class="panel__title">Palimpsest &rarr; context</span></div>
    <div class="panel__body"><!-- a .stratum --></div>
    <div class="panel__foot">…</div>
  </div>
</div>
```

`.chunk` is a retrieved passage as a similarity search would hand it over. `.chunk__score` is for
a short right-aligned label. Do not put invented cosine numbers in it — the point is that the
ordering carries no time, so `match` on every chunk makes the argument more honestly than
fabricated scores.

### 6.3 Cards

Flat, hairline, a 2px top rule. Use `.grid-2` / `.grid-3` around them. `<a class="card">` works.

```html
<div class="grid-3">
  <div class="card">
    <p class="card__title">Short claim</p>
    <p class="card__body">Two or three lines of evidence.</p>
  </div>
  <a class="card" href="field.html">
    <p class="card__title">A card that links</p>
    <p class="card__body">Hovers to an amber top rule.</p>
  </a>
</div>
```

### 6.4 Data tables

**Always** inside `.table-wrap`, which scrolls on its own so the page never scrolls sideways.
`site.js` adds `tabindex="0"` and `role="region"` to any wrapper that actually overflows, so it
is keyboard scrollable — you do not add those yourself.

- Numeric cells: `class="num"` → right-aligned, mono, tabular figures.
- First column: `<th scope="row">`, usually `class="mono"` for system names.
- Long text cells: `class="wrap-cell"` (cells are `nowrap` by default).
- Our own row: `class="is-ours"` on the `<tr>` — amber wash and a tick. **One row per table.**
- Signed deltas: `.pos` / `.neg`, and only there.
- Use `<caption>` for the exact scope of the measurement (dataset, n, haystack).

```html
<div class="table-wrap">
  <table class="data">
    <caption>LongMemEval, all 6 categories, 470 questions, all judged, oracle haystack</caption>
    <thead>
      <tr>
        <th scope="col" class="wrap-cell">system</th>
        <th scope="col" class="num">accuracy</th>
        <th scope="col" class="num">95% CI</th>
        <th scope="col" class="num">tokens</th>
      </tr>
    </thead>
    <tbody>
      <tr class="is-ours">
        <th scope="row" class="mono">palimpsest</th>
        <td class="num">0.589</td>
        <td class="num">[0.544, 0.633]</td>
        <td class="num">949</td>
      </tr>
      <tr>
        <th scope="row" class="mono">hybrid_rag</th>
        <td class="num">0.553</td>
        <td class="num">[0.508, 0.598]</td>
        <td class="num">936</td>
      </tr>
    </tbody>
  </table>
</div>
```

### 6.5 Code

No syntax highlighter. Optional hand-applied spans: `.c` comment, `.k` keyword, `.s` string.
Keep the `<pre><code>` on one line with its content — leading whitespace is significant.

```html
<div class="code">
  <div class="code__label">python</div>
<pre><code><span class="k">from</span> palimpsest <span class="k">import</span> Memory

mem = Memory()
mem.recall(<span class="s">"Where do I live?"</span>)
<span class="c"># current value only</span></code></pre>
</div>
```

Inline code inside `.prose` is just `<code>`; it is styled automatically.

### 6.6 Callouts

```html
<div class="callout callout--caveat">
  <p class="callout__title">Read this with the number</p>
  <p>In both LongMemEval tables the confidence interval <strong>overlaps the runner-up</strong>,
     so the margin over second place is <strong>not statistically significant</strong>. The margin
     over BM25 and below is. And we lose LoCoMo.</p>
</div>

<div class="callout callout--loss">
  <p class="callout__title">Where it loses</p>
  <p>…</p>
</div>

<div class="callout callout--note">
  <p class="callout__title">Note</p>
  <p>…</p>
</div>
```

`--caveat` is amber-filled and is reserved for the mandatory caveat. `--loss` has a `--c-neg`
edge and is for the categories and datasets we lose. `--note` is neutral.

### 6.7 Buttons, pills, stats

```html
<a class="btn" href="docs.html">Read the docs</a>
<a class="btn btn--ghost" href="benchmarks.html">Benchmarks &amp; caveats</a>
<button class="btn btn--ghost btn--sm" type="button">Small</button>

<span class="pill">neutral</span>
<span class="pill pill--current">current</span>
<span class="pill pill--superseded">superseded</span>
<span class="pill pill--corrected">corrected</span>
<span class="pill pill--accent">recovered</span>

<div class="grid-3">
  <div class="stat stat--accent">
    <span class="stat__value">320</span>
    <span class="stat__label">tests, CI green on Python 3.11&ndash;3.13</span>
  </div>
</div>
```

At most one `.stat--accent` per row.

### 6.8 Text utilities

| class | use |
|---|---|
| `.eyebrow` | mono uppercase kicker above a heading |
| `.lede` | larger intro paragraph |
| `.prose` | article body: 68ch measure, spacing, lists, blockquote, inline code |
| `.meta` | small mono metadata |
| `.meta-note` | small muted note paragraph |
| `.mono` / `.num` | mono + tabular figures (`.num` also right-aligns inside tables) |
| `.muted` | `--c-ink-3` |
| `.pos` / `.neg` | signed deltas only |
| `figcaption` / `.figcaption` | caption under a figure or diagram |
| `.dl-axes` | the valid-time / transaction-time definition list |
| `.list-plain` | unstyled list |
| `.todo` | **a number we do not have.** Dashed red. Never ship a page with one silently |
| `.visually-hidden` | screen-reader-only text |
| `.mt-0 … .mt-6`, `.measure`, `.measure-short` | spacing and measure, instead of inline styles |

---

## 7. `site.js`

Tiny, dependency-free, ~100 lines. It does exactly four things:

1. Applies the stored theme during head parsing (no flash), then wires `[data-theme-toggle]` to
   flip `data-theme` on `<html>` and persist to `localStorage` under `palimpsest-theme`.
   All storage access is `try`/`catch`-wrapped so `file://` never throws.
2. With no stored preference, the site follows the OS. The toggle button's label is the theme it
   will switch *to*, and it carries an `aria-label` saying so.
3. Wires `[data-nav-toggle]` against `.site-header[data-nav]` for the mobile menu, including
   `aria-expanded` and Escape-to-close.
4. Marks overflowing `.table-wrap` elements as focusable scroll regions.

Nothing else belongs in it. No scroll animation, no observers, no analytics.

---

## 8. Accessibility contract

- Landmarks on every page: `header`, `nav[aria-label]`, `main#main`, `footer`; a `.skip-link` first.
- Exactly one `<h1>`; heading levels never skip.
- `:focus-visible` is a 2px `--c-focus` outline with 2px offset, defined globally — do not remove it.
- All body/label text meets WCAG AA in both themes (§2). Controls have a ≥3:1 boundary.
- Colour is never the only signal: superseded layers carry a "superseded" pill and a date range,
  not just amber; our table row carries its name, not just a wash.
- `prefers-reduced-motion: reduce` is honoured globally. There are no autoplaying animations —
  the hero's plate texture is static.
- `alt` on every image. Tables get a `<caption>`, `scope` on all headers.

---

## 9. Information architecture

Seven pages, flat, no nesting. Nav order is the reading order for a skeptical engineer:
what it does → does it work → do we believe anyone's numbers → do we believe yours → how do I use it.

| file | h1 subject | what belongs on it |
|---|---|---|
| `index.html` | the thesis | **Built.** The side-by-side (vector store vs. Palimpsest), the mechanism in brief, the headline result with its caveat, the field problem in brief, install, who built it. Everything links onward. |
| `how-it-works.html` | mechanism | The claim ledger and version chains; valid vs. transaction time in depth; supersession vs. correction (use `.stratum` with `.is-corrected`); the three retrieval tiers; open-world predicate canonicalization — the cosine table, shortlist→adjudicate→veto, and why there is no predicate whitelist. |
| `benchmarks.html` | the measurements | The harness first (same process, same answering model, judge separate from answerer, unmodified standard Mem0 judge prompt, same 1,024-token budget, identical extracted claims). Then all four tables: LongMemEval all-6 with CIs, per-category, LongMemEval-S knowledge-update, distractor robustness, LoCoMo. The caveat callout, and the LoCoMo loss stated plainly, not buried. Steelmanned baselines explained. |
| `field.html` | the reproducibility problem | Self-reported vs. measured; MemDelta (arXiv 2606.29914); the LightMem reproduction (arXiv 2607.29104); the wrong LoCoMo category labels in the Mem0 → Memobase → Backboard lineage; the deprecated 2025-09-19 LongMemEval release; no public leaderboard exists. Verified from primary sources — cite each one. |
| `audit.html` | our own defects | The nine defects found in this engine, including the one where it could see the future and inflate its own results, and the one whose fix *lowered* our LoCoMo score. This page is the credibility of the whole site; write it as a log, not an apology. |
| `docs.html` | using it | Install, the three dependencies, the `Memory` API (`ingest`, `recall`, `recall(as_of=)`, `timeline`), storage notes (SQLite, Postgres-portable schema, nothing deleted), and "what this is not". |
| `about.html` | what it is and who stands behind it | Independent, Apache-2.0, nothing for sale — never implied to be a company or a team, no "we", no fake org. Deliberately no age or location: the measurements are the argument, and personal detail on a page pushed to HN/Reddit/X is a locator, not a credential. Status: alpha, the API may change. |

Cross-linking rules: any headline number links to `benchmarks.html`; any claim about the field
links to `field.html`; any claim about our own honesty links to `audit.html`.
