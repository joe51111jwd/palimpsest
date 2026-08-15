# Sending checklist

Read this before sending anything in `DRAFTS.md`. The drafts are written to be sent by
James, by hand, from his own account. No agent, script, or MCP tool sends any of them.

---

## 0. Hard stops — guardian required

Any of the following ends the email thread and moves to a conversation with James's
guardian **before** replying:

- money in any direction — payment, a bounty, a grant, an honorarium, "we'd like to sponsor"
- a contract, letter of intent, consulting agreement, or CLA that isn't a plain OSS CLA
- **an NDA, or any request to share results/code "confidentially" or under embargo**
- a job offer, internship with paperwork, or equity
- anything asking for a legal name + address + tax form (W-9, W-8BEN)
- a company or lab asking to license, resell, or white-label Palimpsest

Safe to answer alone: technical questions, "which file did you run", "send the harness",
"open a PR", "can I cite this", a call to talk about the results with no paperwork attached.
If in doubt, the reply is: *"I'm 16 — anything with a document attached needs my guardian to
look at it first. Happy to keep talking about the technical side meanwhile."*

---

## 1. Before the first send (once)

- [ ] Repo is public, CI green, README results table matches every number quoted in the drafts.
- [ ] `docs/LANDSCAPE.md` actually contains the LoCoMo category-mapping table — emails #4 and
      #5 point at it by name.
- [ ] `docs/AUDIT.md` contains the nine defects — emails #2 and #8 cite the count.
- [ ] `docs/RESULTS.md` shows the LoCoMo loss on the same page as the LongMemEval win.
- [ ] The `zep_style` and `mem0_style` files each carry a header comment saying they are
      re-implementations and not the vendors' systems. Emails #6, #7, #9 promise this.
- [ ] `./scripts/fetch_data.sh` + the `python -m bench.run` line in the README run clean from a
      fresh clone. Someone will try it.
- [ ] Sending address is `atwbusinessjames@gmail.com` (or a cleaner personal address), with a
      real display name. Not a school account.
- [ ] Signature is three lines max: name, "high school sophomore, Las Vegas", repo URL.

## 2. Before each individual send

- [ ] **Re-verify the address on the day.** Recon captured these 2026-08-13/14. Open the
      source again: Di Wu's homepage mailto, ielab author block in arXiv 2607.29104, Adyasha's
      GitHub profile, Bansal's UNC faculty page, Pavlo's GitHub profile, Deshraj's GitHub
      profile. If the address moved, use the new one; if it disappeared, don't guess.
- [ ] **Re-verify the fact you're leading with.** Specifically: Di Wu's README news log still
      says what #1 quotes; issues #44–#53 still exist; `ielab/Reproducing-LightMem` still shows
      89.0 vs 77.7; the LoCoMo repo's last push date; the LoCoMo_refined agreement numbers.
      A stale quoted fact is worse than no email.
- [ ] Every number in the email matches the current README. If a rerun changed a number,
      update the draft before sending, not after.
- [ ] The email says "I" and never "we".
- [ ] The email states the LoCoMo loss and the overlapping CI. Both, in every one.
- [ ] There is exactly one ask, and it's answerable in a sentence.
- [ ] Plain text. No attachments, no images, no tracking pixel, no link shortener, no calendar
      link, no "quick 15 minutes?".
- [ ] Not BCC'd or CC'd to anyone outside that person's own team. Never mail two of these
      people on the same thread.
- [ ] Subject line is the one in the draft — specific, no "quick question", no emoji.
- [ ] Read it out loud once. If a sentence sounds like marketing, delete the sentence.

## 3. Order and pace

**Max 2 sends per day. Max 5 per week.** These are people who talk to each other; a burst
looks like a campaign and gets treated like one.

| Day | Send | Notes |
|---|---|---|
| 1 | #1 Di Wu | Highest value and most likely to reply. Send this first and see how the framing lands before spending the others. |
| 1 | #2 Shuai Wang (cc ielab) | Independent of #1, different community. |
| 3 | #4 Adyasha Maharana | Leads with the loss; safest of the vendor-adjacent ones. |
| 4 | #5 Mohit Bansal | One day after Adyasha, and it says so in the text. |
| 8 | #6 Pavlo Paliychuk | Only after the `zep_style` file header is in place. |
| 10 | #7 Deshraj Yadav | Only after the `mem0_style` file header is in place. |
| 11 | #8 Dell Zhang (LinkedIn) | Not email; connection request first, message if accepted. |
| 17 | #9 Taranjeet Singh | Only if #7 has had ≥7 days. Skip entirely if Deshraj replied — one Mem0 thread is enough. |
| 15+ | #3 Guido Zuccon | **Only** if #2 is silent after 14 days. |

Rules that override the table:

- If #1 replies with a correction, **stop and fix it before sending anything else.** A
  corrected number quoted in five more emails is unrecoverable.
- If any recipient says the harness has a flaw, pause the queue until it's fixed or
  understood. Sending the same claim to the next person after being told it's wrong is the
  one thing that ends this permanently.
- Never send #6 and #7 the same day. Zep and Mem0 read each other's mentions.

## 4. Follow-ups

- **One** follow-up, after 10–14 days, three sentences max: restate the single ask, note that
  no reply is a fine answer, done. No second follow-up ever.
- No follow-up at all to #5 (professors), #8, or #9.
- If someone forwards you to a student or a colleague, that's a fresh first email to a new
  person — go back to section 2 and verify their address too.

## 5. Handling replies

**A correction to our numbers** — best possible outcome. Reply within 24h, thank them, fix
it, rerun, and tell them what changed. If it lowers our score, publish it anyway and say who
caught it. Add them to the README's acknowledgements if they're willing.

**"Your baseline misrepresents my system"** (#6, #7, #9) — agree immediately, ask for the one
change that would fix it, and offer to rename the baseline or pull it until fixed. Do not
defend the number. The baselines are labelled re-implementations precisely so this is cheap
to concede.

**"Send the harness / open an issue / open a PR"** — do it the same week, while attention
lasts. Keep the PR minimal and self-contained.

**"Which dataset version did you use?"** — answer exactly, with the file hash if possible.

**Hostile or dismissive** — one polite line acknowledging it, then stop. Do not argue in
public threads, do not subtweet, do not respond to a hostile GitHub comment more than once.

**Press, podcast, or "can I write about this"** — fine to say yes to being written about, but
send the link to the repo and let the numbers speak; do not do a call without telling a
parent, and route anything with a contract to section 0.

**Anything from section 0** — reply "thanks, I need to loop in my guardian on this one," then
actually do it before replying further.

## 6. Log

Fill this in as you send. Two-word status is enough.

| # | Person | Sent | Follow-up sent | Reply | Outcome |
|---|---|---|---|---|---|
| 1 | Di Wu | | | | |
| 2 | Shuai Wang | | | | |
| 3 | Guido Zuccon | | | | |
| 4 | Adyasha Maharana | | | | |
| 5 | Mohit Bansal | | | | |
| 6 | Pavlo Paliychuk | | | | |
| 7 | Deshraj Yadav | | | | |
| 8 | Dell Zhang | | | | |
| 9 | Taranjeet Singh | | | | |
