# Palimpsest — site

The public site for [palimpsest-memory](https://pypi.org/project/palimpsest-memory/).
Plain static HTML, CSS and JavaScript. No build step, no framework, no `npm install`,
no external requests of any kind. It works offline and it works from `file://`.

Design system and page-authoring rules: [`DESIGN.md`](DESIGN.md). Read that before editing
any page. `assets/site.css` and `assets/site.js` are shared and are not edited per-page.

---

## Contents

| Path | Page |
|---|---|
| `index.html` | Front page |
| `how-it-works.html` | The interval model, predicate canonicalisation, the guards |
| `benchmarks/index.html` | Every result table, including the benchmark this engine loses |
| `benchmarks/methodology.html` | The harness, and what makes a number comparable |
| `leaderboard/index.html` | Self-reported vendor claims next to measured re-runs |
| `leaderboard/SUBMISSIONS.md` | How to submit a system to that board |
| `field.html` | Why published agent-memory numbers do not reproduce |
| `audit/index.html` | Nine defects found in this engine |
| `docs/index.html` | Install and usage |
| `docs/api.html` | API reference |
| `about.html` | Who built this |
| `404.html` | Not-found page (see the note below) |

Support files: `assets/site.css`, `assets/site.js`, `robots.txt`, `sitemap.xml`,
`vercel.json`, `DESIGN.md`, this README.

---

## Preview locally

```sh
cd site
python3 -m http.server 8000
```

Then open <http://localhost:8000/>.

Nothing else is required — there is no dependency to install and nothing to compile.
Every page also opens correctly by double-clicking the `.html` file directly, with one
deliberate exception described below.

To sanity-check every page and every link before you push:

```sh
cd site
python3 -m http.server 8000 &
for p in / /index.html /how-it-works.html /field.html /about.html /404.html \
         /benchmarks/index.html /benchmarks/methodology.html \
         /leaderboard/index.html /audit/index.html \
         /docs/index.html /docs/api.html; do
  printf '%3s  %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8000$p")" "$p"
done
kill %1
```

### The one page that needs a web root

`404.html` references `/assets/site.css` and `/assets/site.js` with **root-absolute**
paths, and its links are root-absolute too. This is deliberate: Vercel serves that one
file in response to a request for *any* unmatched path, at any depth (`/nope`,
`/docs/nope`, `/a/b/c`), so a relative path would resolve differently every time and be
wrong most of the time. Root-absolute is the only correct choice for a not-found page.

Its links keep the `.html` extension (`/benchmarks/index.html`, not `/benchmarks`) so the
page works under a plain static server as well as under Vercel's `cleanUrls`, where the
`.html` form 308-redirects to the clean one.

The consequence is that double-clicking `404.html` from the filesystem shows unstyled but
still readable and fully navigable HTML. Served from a web root it is styled like every
other page. No other page has this property.

---

## Deploy to Vercel

The site is the `site/` subdirectory of the repository, so the Vercel project's
**Root Directory** must be set to `site`. Without that, Vercel sees `pyproject.toml`
at the repository root and tries to build a Python project.

`vercel.json` in this directory sets:

- `framework: null`, `outputDirectory: "."` — serve these files as they are, no build.
- `cleanUrls: true` — `/about.html` is served at `/about`; the `.html` URL 308-redirects
  to it. All in-page links keep their `.html` extension so the pages still work from
  `file://`; the redirect makes the canonical URL the clean one.
- `trailingSlash: false`.
- Security headers on every response, including a strict `Content-Security-Policy`.
  The policy is `default-src 'self'` with no `unsafe-inline` anywhere, which is only
  possible because there is not one inline `<script>` or `style=` attribute in the
  entire site. Adding either will break the site under this header — don't.
- Caching. `/assets/*` gets `max-age=600, s-maxage=31536000, stale-while-revalidate=86400`:
  the CDN holds it for a year, and Vercel purges the CDN on every deployment, so a
  deploy still ships instantly. HTML gets `max-age=0, must-revalidate` so page edits are
  never served stale.
  `assets/site.css` and `assets/site.js` have **no content hash in their filenames**, so
  `immutable` is deliberately not used. If you ever want immutable asset caching, add a
  hash to the filenames first.
- `.md` files (`leaderboard/SUBMISSIONS.md`) are served as `text/plain` so they render in
  the browser instead of downloading.

### First deploy

```sh
cd site
vercel link          # choose or create the project, set Root Directory to `site`
vercel               # preview deployment
vercel --prod        # production
```

Or connect the GitHub repository in the Vercel dashboard, set Root Directory to `site`,
leave Framework Preset as "Other", and leave the Build Command empty.

### Before the first production deploy

`robots.txt` and `sitemap.xml` both hard-code the origin
`https://palimpsest-memory.vercel.app`, which is a **placeholder** — no production domain
has been chosen. Replace it in both files with the real origin once it exists:

```sh
cd site
sed -i '' 's|https://palimpsest-memory.vercel.app|https://REAL-ORIGIN|g' robots.txt sitemap.xml
```

`sitemap.xml` lists the ten real pages at their clean URLs. `404.html` is intentionally
not listed and carries `<meta name="robots" content="noindex">`.

---

## House rules for editing

These come from `DESIGN.md` and from the project's stated facts. They are not style
preferences.

1. Do not edit `assets/site.css` or `assets/site.js` from a page. Build against the
   existing classes; if one is missing, say so rather than adding a `<style>` block.
2. No inline `style=` attributes and no inline `<script>`. The CSP forbids them.
3. No external requests. No font CDNs, no analytics, no remote images.
4. One `<h1>` per page, in the hero. Sections are `<h2>`.
5. Every number must match the project's measured results exactly. If a number is not
   backed by a file in `results/`, do not invent it — leave a `<span class="todo">`.
6. **The caveat travels with the headline.** Wherever a LongMemEval headline number
   appears, the `.callout--caveat` stating that the confidence interval overlaps the
   runner-up appears on the same page near it. Same for LoCoMo: the engine loses it, and
   the page says so.
7. Tables always go inside `.table-wrap`.
8. Check both themes and a 375px viewport before calling a page done.
