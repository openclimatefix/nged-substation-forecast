# Brief: find an open-access copy of one paper

Fill in the bracketed fields and hand the whole file to a sub-agent. Delete nothing else. The
constraints and the reporting format are what make the answer usable, and a brief that drops them
comes back as an unverifiable "I couldn't find it".

Read [`SKILL.md`](SKILL.md) under "Assembling the library" first if you are the one running the
search rather than delegating it. The brief below is the same knowledge, arranged for someone who
has no other context.

---

## The paper

- **Citation**: [authors, year, title, venue]
- **Digital object identifier**: [doi]
- **Authors and their affiliations**: [names; and whether they are at universities or in industry,
  which decides half the routes below]
- **What the abstract already tells us**: [the claims we can already make]
- **What we need from the full text**: [the specific numbers — the baseline, the normalisation, the
  horizon, the split. Name them, so the agent knows what to extract and what to say it could not
  find.]

## Why the answer matters

[Who publishes the review, who reads it, and what a wrong claim costs. Say explicitly that a
well-evidenced negative is as useful as a find, because it lets us either cite the abstract honestly
or decide to pay.]

## The two kinds of route

**The first kind indexes papers.** Sweeping the whole of the first kind costs about a minute, and
tells you only whether an *indexed* copy exists:

- Unpaywall — `https://api.unpaywall.org/v2/<doi>?email=<address>`
- OpenAlex — `https://api.openalex.org/works/doi:<doi>`; read the full `locations` list, not just
  `best_oa_location`
- Semantic Scholar —
  `https://api.semanticscholar.org/graph/v1/paper/DOI:<doi>?fields=title,openAccessPdf,externalIds`
- arXiv, CORE, Crossref, OpenAIRE, Zenodo
- the institutional repository of any university named in the affiliations

**A "closed" answer from every one of those is a statement about indexed routes, not about the
paper.** They all work by indexing, so a copy on somebody's own web page is invisible to all of them
by construction. One paper was recorded as unobtainable after every route above came back empty, and
the full text was on a co-author's personal website throughout. **Treat the first sweep as the
opening move, never as the search.**

**The second kind is indexed by nobody, and is where the copy usually turns out to be:**

- **Each author's own web page.** Search every author's name separately, not only the first
  author's. This is the route that has actually worked.
- **The publisher's or the learned society's own portal**, checked directly rather than inferred
  from an aggregator. Two Korean papers reported `closed` by both Unpaywall and OpenAlex were free,
  under a Creative Commons licence, from the society's own journal portal.
- **The employer's website, in the employer's own language**, when the authors work in industry. A
  utility posts conference papers under *Veröffentlichungen* or *Publikationen*, not "publications".
- **The authors' conference slides**, which sometimes carry the number the paper puts behind a
  paywall.
- **The venue's own repository or national committee**, checked for whether it covers the paper's
  year at all before concluding anything from an empty result.

## Traps that produce a confident, wrong negative

- **The namesake.** An ORCID and Zenodo trail under the right surname belonged to a researcher in a
  different field entirely. That false trail is what made a wrong negative look convincing. Confirm
  a profile belongs to the right person — by affiliation, by co-authors, by field — before
  concluding anything from what it lists.
- **A route that cannot answer.** A bare domain that does not resolve, a results page rendered in
  JavaScript, and a mistyped path all print exactly what a genuine absence prints. One venue
  repository does not resolve without the `www` prefix.
- **A route that stays silent versus a route that says no.** An author profile that loads, lists
  that author's other papers, and does not list this one is real evidence. An empty search box that
  returned a 403 is not evidence of anything. Say which kind each of your results is.

## Hard constraints — absolute

- **No browser tools of any kind.** Not `mcp__Claude_Browser__*`, not `mcp__claude-in-chrome__*`. A
  browser tool once opened a file-download dialog on an unattended screen with nobody watching.
  Command line only: `curl`, `grep`, `sed`, `awk`, `python3`, `pdftotext`.
- **No pirate mirrors.** Sci-Hub, LibGen, Anna's Archive, Z-Library and anything of that kind are
  forbidden regardless of what you find or how convenient it would be.
- **Never bypass a bot challenge or CAPTCHA.** A gated route is a closed route. Report the URL so a
  person can open the page themselves, and move on. Publisher portals, ResearchGate, LinkedIn, and
  several university search interfaces all refuse a command-line client. A 403 from any of them says
  nothing about whether the paper is open behind it.
- **Do not spawn sub-agents.** A grandchild agent cannot report back once its parent has finished.
- **Do not pay for anything**, and never enter credentials or personal details anywhere.
- **Do not save the PDF into the repository** and **do not edit any file in the repository**.
  Verifying a download under `/tmp` is fine; report the URL and let the caller place the file.

## What to report

For every route: the exact URL or API call, the HTTP status or the relevant slice of the response,
and what the response told you. **Prove each search ran** — quote the field values you actually
read, and give a control hit count from a string you know is present. A "no results" from a call you
cannot show returned a valid response is not a finding.

Then either:

**Found** — the direct URL; proof the copy is open (status, content type, size); confirmation from
the first page that the title and the authors are the right ones; the licence if stated; and whether
the copy differs from the version of record, since an author copy sometimes carries a different
title. Then extract everything listed under "What we need from the full text" above, and say plainly
which of those items the paper does not state.

**Not found** — every route tried with its outcome, split into routes that answered no and routes
that could not answer; which routes were bot-gated, with the URL a person could open; and an honest
assessment of whether any plausible open route remains unexplored.

Do not guess, and do not fill a gap with plausible-sounding detail. "I could not determine this" is
a good answer; an invented figure in a published review is the worst outcome there is.
