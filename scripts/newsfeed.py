import anthropic
import httpx
import nh3
import re
import smtplib
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# The report is built from live web-search results, which are untrusted input.
# A malicious page can attempt indirect prompt injection to make the model emit
# tracking beacons (<img>), scripts, or javascript:/data: links that would fire
# or exfiltrate when the email is opened. We never trust the model output as
# safe HTML — it is run through an allowlist sanitizer before being emailed.
# Only these tags/attributes survive; everything else (img, script, style,
# iframe, event handlers, non-http(s)/mailto URLs) is stripped.
#
# `span` and `class` are allowed so the model can mark up structural hooks
# (item cards, tier headers, pill tags) that our trusted stylesheet targets.
# They are cosmetic only and don't widen the security surface: `class` cannot
# execute or exfiltrate, and the `style` attribute, `<style>` element, scripts,
# images, and event handlers all remain stripped. Worst case from an injected
# class is a misplaced pill — a visual nuisance, not a vulnerability.
ALLOWED_TAGS = {
    "h2", "h3", "p", "strong", "em", "ul", "ol", "li", "div", "a", "br", "span",
}
_CLASS_ONLY = {"class"}
ALLOWED_ATTRIBUTES = {
    "a": {"href", "title", "class"},
    "div": _CLASS_ONLY,
    "span": _CLASS_ONLY,
    "p": _CLASS_ONLY,
    "h2": _CLASS_ONLY,
    "h3": _CLASS_ONLY,
    "ul": _CLASS_ONLY,
    "ol": _CLASS_ONLY,
    "li": _CLASS_ONLY,
    "strong": _CLASS_ONLY,
    "em": _CLASS_ONLY,
}
ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}


def sanitize_html(html):
    return nh3.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        url_schemes=ALLOWED_URL_SCHEMES,
    )


# Example profile only. The real candidate profile is injected at run time via
# the CANDIDATE_PROFILE environment variable (a GitHub Actions secret) so
# personal details never live in the repository. A custom profile must keep
# this shape — a "WHO THE CANDIDATE IS" section (background, target roles and
# verticals, home metro area, remote preference) followed by a "WATCHLIST
# COMPANIES" section ending in the company list — because the prompt text
# that follows it refers back to both. It may optionally add a "BODY OF WORK"
# section after the watchlist; the prompt uses it to surface builder-post
# seeds (news that intersects what the candidate has built) and skips that
# seeding if it is absent.
DEFAULT_PROFILE = """WHO THE CANDIDATE IS

The candidate has 12+ years of experience building and leading risk functions at high-growth technology companies. They are looking for Director, Senior Director, or VP level GRC or Technology Risk or Chief Risk leadership roles at technology-forward companies in regulated verticals. The filter prioritizes regulatory surface over industry vertical: fintech, crypto, healthtech, AI, enterprise SaaS with government contracts, life sciences, financial services, consumer platforms with significant privacy exposure, and defense-adjacent technology all fit the profile. Within these verticals, companies that have received enforcement actions, consent orders, or significant regulatory attention are higher-priority targets, but any company in a regulated vertical is in scope.

The candidate is remote-based in the San Francisco Bay Area; treat that as their home metro area. They prefer fully remote roles but will consider hybrid for the right local opportunity.

---

WATCHLIST COMPANIES

Stripe, Plaid, Block, Robinhood, Oscar Health, Databricks, Anthropic, OpenAI, Uber.

---

BODY OF WORK

The candidate builds rather than only audits: they maintain in-house quantitative risk tooling grounded in FAIR, write their own runbooks, and prototype automation. Their recurring points of view, which their LinkedIn content returns to, include risk as an engineering discipline, quantification over qualitative heat maps, and second-line functions that drive business decisions rather than produce paperwork. Use this section to surface builder-post opportunities — news items that intersect this body of work."""


# Approximate published Opus 4.8 rates ($5/$25 per MTok), in USD per token.
# Adjust if pricing changes — these only drive the logged cost estimate, not
# anything functional.
PRICE_INPUT = 5 / 1_000_000           # fresh (uncached) input
PRICE_CACHE_WRITE = 6.25 / 1_000_000   # cache creation = 1.25x input
PRICE_CACHE_READ = 0.5 / 1_000_000     # cache read = 0.1x input
PRICE_OUTPUT = 25 / 1_000_000
PRICE_WEB_SEARCH = 10 / 1_000          # $10 per 1,000 searches


def accumulate_usage(totals, usage):
    """Add one API response's usage onto the running totals for the run."""
    server_tool = getattr(usage, "server_tool_use", None)
    totals["input"] += getattr(usage, "input_tokens", 0) or 0
    totals["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    totals["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    totals["output"] += getattr(usage, "output_tokens", 0) or 0
    totals["searches"] += (
        getattr(server_tool, "web_search_requests", 0) or 0 if server_tool else 0
    )


def log_usage(totals):
    """Print run-total token/search usage and an estimated dollar cost."""
    est_cost = (
        totals["input"] * PRICE_INPUT
        + totals["cache_write"] * PRICE_CACHE_WRITE
        + totals["cache_read"] * PRICE_CACHE_READ
        + totals["output"] * PRICE_OUTPUT
        + totals["searches"] * PRICE_WEB_SEARCH
    )

    print(
        f"Usage across {totals['api_calls']} API call(s) — "
        f"input(fresh): {totals['input']:,}, cache write: {totals['cache_write']:,}, "
        f"cache read: {totals['cache_read']:,}, output: {totals['output']:,}, "
        f"web searches: {totals['searches']:,}\n"
        f"Estimated cost: ${est_cost:.2f} "
        "(rate estimate; verify against Anthropic pricing)"
    )


# Upper bound on pause_turn continuations (the server-side tool loop pauses
# roughly every 10 tool iterations); a guard against a runaway loop, not a
# budget — search spend is capped by max_uses on the web_search tool. Raised
# from 8 to 12 alongside the higher search cap: more searches plus per-posting
# fetches plus content stake-testing means more tool iterations, and the run
# must have room to finish before this guard trips.
MAX_PAUSE_CONTINUATIONS = 12

# Full-scan retries when the streaming connection dies mid-read ("peer closed
# connection", read timeout). The SDK's max_retries doesn't cover these — it
# only retries failed request setup — and a dropped stream loses the whole
# in-flight report, so the only recovery is to start the scan over.
STREAM_RETRIES = 2


def get_newsfeed():
    # Weekly cadence: one transient 529/5xx would otherwise cost a whole week,
    # so retry harder than the SDK default of 2.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=4)

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%B %d, %Y")

    profile = os.environ.get("CANDIDATE_PROFILE", "").strip() or DEFAULT_PROFILE

    prompt = f"""Today's date is {today}. The scan window is {week_ago} through {today}. Only include items published within this window. Verify the publication date of every source before including it — reject anything outside the scan window.

You must use live web search for every item in this report. Do not rely on your training data for any factual claim, company development, or regulatory action. If you cannot find a live, dated source for an item, do not include it. You also have a web_fetch tool: use it to open pages directly whenever a rule below requires verifying what a page actually contains — especially job postings, which must be fetched and confirmed live, never trusted from search snippets. Only link to the original source — the actual article, SEC filing, or press release — not to aggregators or search result pages. Try to avoid sites whose content is behind a paywall.

If a category yields no confirmed items this week, output the category header followed by: "Nothing confirmed this week." Do not fill empty categories with soft sources or loosely relevant items.

You are a research assistant supporting a senior GRC and Technology Risk executive — referred to throughout as "the candidate" — who is actively searching for a Director to VP level role. You also identify content opportunities for the candidate's LinkedIn presence, where they post industry analysis aimed at CISOs and executive talent partners.

Only cite sources from original publications — official regulatory filings, company press releases, or established news outlets (Reuters, Bloomberg, WSJ, TechCrunch, SC Media, Dark Reading), plus GRC Report (grcreport.com), a governance-risk-compliance trade publication with its own newsroom that ranks alongside the security trade press above. Actively scan GRC Report's news and insights feeds as a discovery source, especially for the content categories (4 through 7). Also draw directly from: FDIC and OCC enforcement action databases, SEC EDGAR full-text search for cybersecurity disclosure filings, and FinCEN for anything touching crypto or financial services clients. Reject aggregator sites, content farms, or any URL you cannot confirm resolves to a real, dated article. When GRC Report or any secondary outlet reports on a primary event (a regulatory action, filing, or company announcement), still link the primary source in Categories 0 through 3 where one exists; cite GRC Report directly for its own analysis and reporting.

---

{profile}

This watchlist is a starting point, not a boundary. The search is profile-driven, not list-driven: any tech-forward company in a regulated vertical is in scope — fintech, payments, lending, banking-as-a-service, crypto, insurtech, healthtech and digital health, telehealth, AI labs and AI infrastructure, enterprise SaaS with government contracts or FedRAMP exposure, life sciences technology, financial services, consumer platforms with significant privacy exposure, proptech, defense-adjacent technology — regardless of whether it appears above. Companies with recent enforcement actions or regulatory attention are higher-priority, but regulated-vertical membership alone is sufficient to include a company. Expect most of the best findings each week to come from companies NOT on the watchlist.

The profile above may include a BODY OF WORK section. Use it to spot builder-post opportunities — news items that connect to something the candidate has actually built and could anchor a post. It is a discovery aid only: this report names and analyzes positions in neutral, third-person analyst language and never drafts the candidate's post copy or imitates their writing voice. If the section is absent from the profile, skip builder-post seeding.

---

SCAN CATEGORIES AND PRIORITY ORDER

Category 0 and Categories 1 through 3 are job search signals and take priority over Categories 4 through 7, which are content opportunities. Within each tier, items involving named target companies rank above general market developments. Category 8 sits outside both tiers: it is a standing scan for speaking and guest opportunities, neither a job-search signal nor a post-topic content item.

CATEGORY 0 — OPEN ROLES (highest priority, run first)

ROLE TITLES IN SCOPE. Search for Director, Senior Director, VP, Head of, and Chief level roles across the full family of titles this function goes by, not just the literal string "GRC": Governance, Risk & Compliance; GRC; Technology Risk; Enterprise Risk; IT Risk; Information Security Risk; Security GRC; Risk Management; Operational Risk (technology-focused); Security Assurance; Security Compliance; Trust (as in Head of Trust); Third-Party Risk; AI Governance or Responsible AI; Chief Risk Officer; Business Information Security Officer (BISO). At startups and growth-stage companies, "Head of X" is typically Director-to-VP equivalent — treat it as in scope. Compliance-titled roles are in scope when the mandate is technology, security, or AI compliance rather than purely legal/financial compliance.

DISCOVERY STRATEGY. The search is profile-driven: most qualifying roles each week will be at companies not on the watchlist, so do not simply iterate the watchlist company by company. Run these discovery passes, in this order:

1. Watchlist and signal-driven companies: check careers pages of watchlist companies, plus any company surfaced this week in Categories 1 through 3 (enforcement action, consent order, IPO filing or announcement, late-stage funding, new CISO or CRO in seat). A company that just hired a security or risk executive, took an enforcement action, or filed an S-1 is the single strongest predictor of an open risk-leadership search.
2. ATS-wide title sweeps: run site-restricted web searches for the role titles above directly across the major applicant-tracking-system domains — boards.greenhouse.io, jobs.lever.co, jobs.ashbyhq.com, myworkdayjobs.com, jobs.smartrecruiters.com, apply.workable.com — for example: site:boards.greenhouse.io "Director" "Technology Risk". This is the highest-yield way to find companies the candidate has never heard of. Filter the hits to companies matching the regulated-vertical profile.
3. Job aggregators for discovery: LinkedIn Jobs, Built In (the candidate's home metro and remote), Wellfound, and Welcome to the Jungle/Otta. Aggregators are for discovery only — always follow through to the underlying company posting and cite that as the Source, never the aggregator page.
4. IPO pipeline: scan recent S-1 filings on SEC EDGAR and credible IPO-pipeline coverage for regulated-vertical companies approaching public markets within 18 months — these are high-signal hiring windows where GRC investment is most active — and check those companies' careers pages.

Aim for breadth of companies over exhaustive depth on any one company. A weekly report that surfaces 8 to 15 verified roles across many companies is more useful than 3 roles from the watchlist plus an exhausted search budget.

PRESENT ROLES IN TWO TIERS. Search broadly, but do not present the results as one flat list — breadth is valuable for discovery but creates noise when every role is shown with equal weight. Split this category's confirmed roles into two labeled sub-sections, strongest first within each:

- "Strong fits": roles where title/function, seniority (Director, Senior Director, VP, Head of, or Chief level), AND vertical all clearly match the candidate's profile, and the location is the candidate's home metro area or fully remote. These are the roles they should look at first.
- "Broader — worth a look": real, verified, currently-live roles that are a stretch on one dimension — seniority slightly off (e.g. Senior Manager trending toward Director, or a very large-scope VP), vertical adjacent rather than core, or location out-of-area with an unclear or onsite arrangement. Include these for discovery value, but cap this tier at the 8 strongest; if more than 8 qualify, keep the 8 best fits to the candidate's profile and drop the rest rather than padding the list.

Do not relax the liveness/verification rules for either tier — a role must be fetched and confirmed live to appear in either. Tiering is about ranking what you found, never about lowering the bar for what counts as verified. If a role is a genuine strong fit, it goes in "Strong fits" even if it is the only role this week.

The 7-day scan window above does NOT apply to this category. Roles are governed by whether they are currently live, not by when they were first posted. A still-open role posted three weeks ago is in scope; a role posted yesterday that has already closed is not.

Every posting you include must be currently live and open to applications. Search results and search-engine snippets routinely surface roles that have already been filled or closed, so a search hit is not sufficient evidence that a role is open. Before including any role, open the posting page itself with web_fetch and confirm from the fetched content that it is still accepting applications. A role you did not fetch does not go in the report.

A page that returns successfully is NOT proof the role is live. Closed postings very frequently still "work" but silently redirect to the company's default careers homepage, a job-search index, or a generic "open positions" listing, while the original link continues to resolve. You must confirm that the final page you land on actually displays that exact role — its specific title and description, with an active apply control. If the link instead lands on a careers homepage, a job-search or "open positions" index, a search results page, or a "job not found" / "this position is no longer available" page, the role is dead — exclude it. The link you put in the Source field must point to that live, role-specific detail page, not to a redirect target or a careers landing page.

Reject the posting — do not list it — if any of the following are true: the page does not load or returns an error; the link redirects to or lands on a generic careers page, job-search index, or listing rather than the specific role's detail page; the final page does not display that exact role's title and description with an active apply control; the page states the role is closed, filled, paused, on hold, expired, or "no longer accepting applications"; the listing shows no posting or last-refreshed date; or the posting date is more than 30 days before today. When in doubt, exclude rather than guess: an omitted role is fine, a dead role is the failure mode to avoid.

For each confirmed role, provide the role title, company, the posting or last-refreshed date exactly as it appears on the page, and a direct link to the role-specific posting itself (not a search results page, careers homepage, or job-aggregator listing). While you have the posting open to verify it is live, also capture the stated compensation range if one is present — US postings frequently disclose it under pay-transparency laws — and report it exactly as written. Note whether the company has appeared in any other category this week, as co-occurrence is a strong signal. If the company is in IPO preparation or publicly known to be approaching IPO within 18 months, flag this prominently — it is a high-priority hiring signal. If you cannot confirm a single live role this week, output: "Nothing confirmed this week."

For every confirmed role, note its location and work arrangement (remote, hybrid, or onsite) as stated on the posting. If the role's primary location is outside the candidate's home metro area, additionally flag the company's current work-location posture: whether it has recently announced or enforced a significant Return-to-Office (RTO) mandate, or whether it is genuinely remote-friendly. Base this on dated, verifiable sources — the posting's own remote/location terms, a company announcement, or recent news coverage — and say so briefly if you cannot confirm either way. This flag is informational only: do NOT exclude, downrank, or filter out an otherwise relevant out-of-area role because of an RTO push or because the work arrangement is unclear. The candidate still wants to see these roles; the flag simply tells them what they would be walking into. Roles based in the candidate's home metro area, or explicitly advertised as fully remote, do not need the RTO research.

CATEGORY 1 — REGULATORY ACTIONS

Enforcement orders, consent decrees, new rulemaking, or significant regulatory attention affecting any tech-forward company. Relevant regulators include OCC, CFPB, SEC, FTC, FCC, FDA, FDIC, FinCEN, state-level regulators, and major international regulators, notably in Europe. This scope includes digital-asset and stablecoin regulation as ordinary regulatory items: GENIUS Act developments, stablecoin and market-structure rulemaking, and enforcement touching crypto or digital-asset firms.

CATEGORY 2 — ORG SIGNALS

Layoffs, RIFs, restructuring, IPOs, late-stage funding rounds, bank charter applications, or significant M&A activity at tech-forward companies in regulated spaces. For each item, assign a hiring window temperature: Hot (company is likely actively building the function — e.g., post-enforcement action, post-funding, new CISO in seat), Warm (conditions are favorable but timing is uncertain), Cold (company is mid-restructure or in a hiring freeze), or Avoid (signals suggest the candidate should not prioritize this target right now, with a brief reason).

CATEGORY 3 — LEADERSHIP MOVEMENT

Changes in security and risk leadership at tech-forward companies in regulated verticals: new CISO, CSO, Chief Trust Officer, Head of Security, Head of Trust, CRO, or Head of Risk appointments, and departures or announced exits from those seats. Report security and risk leadership changes here rather than under Org Signals; a company may still appear separately in Org Signals for a distinct corporate event. Targets and watchlist companies first, but a leadership change at any regulated-vertical company is in scope, since a company that just brought in a new security or risk executive is the single strongest predictor of an open second-line search.

For each move, in addition to the standard Category fields below, state: who moved and into or out of which seat; the prior holder and where they went if known; whether a GRC, risk, or AI-governance role is already posted at that company (check the careers page, and cross-list the role in Category 0 if you can confirm it live) or is likely to open on the new executive's typical build timeline; and whether this leader belongs on the candidate's commenting list — a named security leader whose feed reaches other security leaders and whose org is a plausible target or peer. Treat a newly seated CISO as a Hot hiring window unless a freeze or restructure argues otherwise.

Base every move on a dated, verifiable source — a company announcement, the executive's own posted announcement, or established coverage — never on inference from a LinkedIn headline alone. If you cannot confirm the move with a real source, leave it out.

CATEGORY 4 — MAJOR INDUSTRY REPORTS AND DATA RELEASES

Scan broadly across major industry data releases and the technology and cybersecurity think tanks, research centers, and standards bodies listed below. Include a new report, framework update, dataset, or publication only when it carries a clear angle relevant to Governance, Risk, and Compliance or AI Governance — skip purely technical or operational releases with no GRC, risk-quantification, board-governance, regulatory, or AI-governance relevance.

Core data releases: Verizon DBIR, CrowdStrike Global Threat Report, Gartner, SANS, CSA, FAIR Institute publications, Hubbard Decision Research content.

Think tanks and research centers: Center for Security and Emerging Technology (CSET), Institute for AI Policy and Strategy (IAPS), Institute for Security and Technology (IST), Institute for Critical Infrastructure Technology (ICIT), UC Berkeley Center for Long-Term Cybersecurity.

Standards bodies and frameworks: NIST Cybersecurity Framework (CSF), NIST Trustworthy & Responsible AI Resource Center, ISO 27001, CIS Critical Security Controls, ISACA, MITRE Corporation, OWASP, PCI Security Standards Council (PCI SSC), HITRUST. Flag new releases, framework revisions, draft guidance, or notable commentary from these bodies when they bear on GRC or AI governance.

CATEGORY 5 — INCIDENTS AND GOVERNANCE FAILURES

Significant breaches, technology failures, or governance failures at tech-forward companies. Include an incident only when it carries a second-line or governance angle a risk leader could actually speak to: a board or audit-committee oversight gap, a risk acceptance or exception that aged badly, a controls or third-party-risk failure, or a disclosure and incident-response decision worth examining. The breach itself is not the item; the governance lesson is. Skip incidents that are purely technical or purely operational with no second-line, governance, or risk-management angle, however large the breach.

CATEGORY 6 — AI GOVERNANCE

EU AI Act implementation, US regulatory activity on AI, enterprise AI deployment failures, agentic AI risk incidents, AI agent governance frameworks. Prioritize items involving agentic AI risk failures, enterprise AI deployment governance gaps, and regulatory action touching AI deployment in financial services or healthtech specifically. These intersect directly with the candidate's positioning as an AI-native GRC leader.

CATEGORY 7 — GRC METHODOLOGY AND ORGANIZATIONAL DESIGN

Developments in quantitative risk (FAIR, CRQ, Hubbard, ERQI), GRC Engineering movement activity, board and audit committee governance, SEC cybersecurity disclosure rules, CISO mandate and org design trends, supply chain and third-party risk.

CATEGORY 8 — BORROWED-AUDIENCE AND SPEAKING OPPORTUNITIES

A standing scan for places the candidate could reach a security-leadership audience as a guest or speaker — not topics to post about. This serves the current goal of landing one borrowed-audience slot ahead of the fall hiring season. Scan for open calls for proposals, calls for speakers, and calls for panelists from GRC, risk, and security-leadership venues and communities: for example the Executive Women's Forum, the FAIR Institute, ISACA, regional BSides events, and RSA-adjacent or Cloud Security Alliance events. Also flag open guest-contributor calls from established GRC or security newsletters and publications where the call is publicly posted.

Only include a call you can confirm is currently open with a real, dated source: a live CFP or submissions page with a stated deadline, or a published call for contributors. Do not infer that a recurring event is accepting submissions because it ran last year; if you cannot confirm an open call this cycle, leave it out. Podcasts rarely publish booking or guest-application pages, so do not fabricate podcast openings — include a podcast only where there is an explicit, dated public call for guests.

For each opportunity, provide: what it is and the venue or publication; the submission deadline or event date exactly as stated; one line on why it fits the candidate as a GRC and risk-quantification voice; how to submit or apply; and a direct link to the call. This category does not use the signal-type pill and is not eligible for THIS WEEK'S RECOMMENDED POSTS, which is for post topics only.

---

SEARCH BUDGET

Job-search categories (0 through 3) have first claim on the search budget; never starve role discovery or live-verification to fund content work. Content categories (4 through 7) should be efficient on discovery — most content items can be assessed from a single search plus the source itself. Reserve a small share of the budget, roughly ten searches, for stake-testing: confirming that the contrarian position behind a High-strength stake actually holds against the evidence before you recommend a post built on it. Spend that stake-testing budget only on the two or three finalist post candidates, not on every content item. The borrowed-audience scan in Category 8 is low-cost: spend at most a couple of searches against known CFP and event pages, and never let it draw from the job-search budget.

---

OUTPUT FORMAT

Begin your response with the opening HTML tag. Do not narrate your search process, describe your methodology, summarize what you are about to do, or include any preamble or transitional language before the HTML output. The report starts with the HTML — nothing before it.
At the top of the report, flag the three highest-priority items across all categories. Rank by: (1) named target company involvement, (2) open role or job search signal over content opportunity, (3) regulatory action over general market development.

For each item in Categories 0 through 3, provide:
- What happened: one to two sentences, factual and specific.
- Why it matters to the candidate: one to two sentences on the job search or content angle.
- Recommended action: a specific next step and, where relevant, a time window. 
- Signal type: Job search signal, Content opportunity, or Both.
- Hiring window temperature (Categories 0, 2, and 3 only): Hot, Warm, Cold, or Avoid, with a one-sentence rationale.
- Fit (Category 0 only): one sentence stating why this role fits the candidate's profile and the single biggest caveat or stretch (e.g. "Core GRC leadership at a post-enforcement fintech; caveat: onsite NYC with no stated remote option"). This is what lets the candidate skim-accept or skim-reject in one read.
- Location and work arrangement (Category 0 only): the role's location and whether it is remote, hybrid, or onsite. For roles outside the candidate's home metro area, also flag whether the company has a recent Return-to-Office (RTO) push or is remote-friendly, with the basis for that flag. This is informational and never a reason to omit the role.
- IPO status (Category 0 only, if applicable): whether the company is in IPO preparation or approaching IPO within 18 months, with the basis (announced plans, S-1 filing, recent funding, public news).
- Compensation (Category 0 only): the pay range exactly as stated on the posting you fetched, including what it covers (base, on-target earnings, bonus, equity) if specified — e.g. "$220K–$265K base + equity." Many US roles disclose this under pay-transparency laws. If the posting shows no range, write "Not disclosed on posting"; you may add a market estimate ONLY if you find a dated, citable public source (e.g. the company's other current postings, a recent Levels.fyi or comparable data point) and label it clearly as an estimate with that source. Never invent or guess a number from general knowledge.
- Source: direct link to the original article, filing, or job posting.

For each item in Categories 4 through 7, provide:
- What happened: one to two sentences, factual and specific.
- Consensus take: one line stating the obvious read — what most GRC commentators will say about this item this week. Naming the crowded position is what lets the candidate avoid it.
- The stake: the contrarian position available here, stated as a claim in neutral third-person analyst language — one to two sentences naming the position, the consensus it cuts against, and the specific sourced fact (a finding, a number, a clause) that supports it. It must take a side a competent CISO would either strongly agree with or bristle at, and be defensible from the standpoint of someone who has built and run a second-line function. Do NOT write it as a post's first line and do NOT imitate the candidate's writing voice.
- Stake strength: rate High, Medium, or Low based on whether a real, defensible disagreement is available here, or whether this is merely a timely topic with no genuine counter-position. Be honest; a Low is useful signal, not a failure. Drop the item from the report entirely if the only available stake restates the consensus — a newsworthy item with no defensible angle is not a content opportunity for this candidate.
- Builder bridge (include only when it genuinely applies): if this item intersects something in the candidate's BODY OF WORK, state in one line how it could anchor a builder post, with the news as the hook and the candidate's own built work as the payload. Omit this field when there is no real connection; never force one.
- Signal type: Content opportunity, Job search signal, or Both.
- Source: direct link to the original article or filing.

When flagging errors and limitations, apply the following rules throughout the report.
If a source is paywalled or only partially accessible, include the item but add a note in the Source field: "Paywalled — summary based on headline and visible excerpt only. Verify before acting."
If a hiring window temperature assessment in Category 2 or Category 3 is based on a single signal or thin evidence, add a note after the temperature rating: "Low confidence — based on limited signal."
If a job posting in Category 0 cannot be confirmed as currently live and accepting applications by opening the posting page, exclude it entirely. Do not list unverified or stale roles even with a caveat — in this category a wrong listing is worse than an omission.
If two or more sources report the same event with conflicting details, include the item but note the conflict: "Conflicting reports — see sources." and provide both links.
If web search returns no results for a specific target company in a given category, do not infer absence of news. Note it as: "No confirmed results found for [company] this week — coverage may be incomplete."
At the end of the report, include a final section titled THIS WEEK'S RECOMMENDED POSTS. From the week's content items, select the three strongest LinkedIn post candidates and rank them, strongest first. Rank by stake strength — the depth of defensible disagreement available — NOT by how big the underlying story is. A smaller story with a sharp, contrarian, well-grounded position outranks a major headline that carries only the obvious take. If fewer than three items clear Medium stake strength, present only those that do and say so, rather than padding with weak candidates.

For each candidate, provide:
- Register: thinky (industry POV, analytical), builder (a first-person account of something the candidate built or solved, anchored to the news item), or human/leadership (warmer, story-driven). Choose builder whenever the item has a Builder bridge and the candidate's own work is the strongest available payload; builder is the highest-converting register and must not be passed over when a genuine bridge exists.
- The position: the contrarian claim this post would advance, in neutral third-person analyst language — the side it takes, stated so a CISO would either agree or bristle. Do NOT draft it as a post's first line and do NOT imitate the candidate's writing voice.
- The cut: one line naming the consensus take this post cuts against, so the candidate can see at a glance why it will stand out instead of blending in.
- The evidence: the one specific fact, finding, or piece of the candidate's own work that makes the position defensible.

Do not write the full posts — give only the register, the position, the cut, and the evidence for each of the three.

FORMAT AND MARKUP

Output ONLY the report body as an HTML fragment. Do NOT include <!doctype>, <html>, <head>, <body>, <style>, or any CSS — a styling shell is wrapped around your output automatically. Do not set any colors, fonts, or style attributes yourself; the only styling you control is the class names listed below, which hook into that shell. Do not invent other class names or use any class not listed here.

Structure:
- <h2> for category headers (e.g. "CATEGORY 1 — REGULATORY ACTIONS").
- <h3> for item titles (the role title + company, the report headline, etc.).
- Wrap every individual item in <div class="item">…</div>.
- <strong> for field labels (e.g. <strong>Why it matters to the candidate:</strong>).
- Plain prose in <p> tags; lists in <ul>/<li>. Use <a href="…"> for every source link.
- For the three highest-priority items at the very top, wrap that whole block in <div class="highlights">…</div>.
- In Category 0, render the two tier sub-headers as <h3 class="tier">Strong fits</h3> and <h3 class="tier">Broader — worth a look</h3>, each followed by that tier's item divs.

Pill tags — wrap ONLY the short rating values (not the labels) in a styled pill <span>:
- Signal type value: <span class="pill pill-signal">Job search signal</span> (also for "Content opportunity" or "Both").
- Hiring window temperature value, color-coded by rating: <span class="pill temp-hot">Hot</span>, <span class="pill temp-warm">Warm</span>, <span class="pill temp-cold">Cold</span>, <span class="pill temp-avoid">Avoid</span>.
So a line reads: <strong>Signal type:</strong> <span class="pill pill-signal">Both</span>. Leave all other field values as plain text.

Do not use markdown. No inline JavaScript, no images, no tables. Keep nesting shallow and clean.
"""

    totals = {
        "input": 0, "cache_write": 0, "cache_read": 0, "output": 0,
        "searches": 0, "api_calls": 0,
    }

    def run_scan():
        """One full scan: stream the request, following pause_turn continuations.

        The server-side tool loop pauses (stop_reason "pause_turn") after ~10
        tool iterations. A scan this search-heavy will hit that several times,
        so keep continuing the same conversation until the model finishes for
        real. Streaming avoids the SDK's 10-minute non-streaming timeout.
        """
        # Cache the large static prompt. The web-search/web-fetch tool loop
        # makes many model turns within this call (and across pause_turn
        # continuations and scan retries); caching means later turns read the
        # prefix from cache at ~10% the cost instead of reprocessing it.
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }],
        }]
        text_parts = []
        message = None
        # The dynamic-filtering web_search/web_fetch tools run their result
        # filtering as server-side code execution inside a container. When the
        # tool loop pauses (pause_turn) with pending code-execution tool uses,
        # the continuation must be pinned to that same container by passing its
        # id back — otherwise the API rejects the resume with "container_id is
        # required when there are pending tool uses generated by code execution
        # with tools." The container id first appears on the paused response.
        container_id = None

        for _ in range(1 + MAX_PAUSE_CONTINUATIONS):
            stream_kwargs = {
                "model": "claude-opus-4-8",
                "max_tokens": 64000,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "tools": [
                    # max_uses caps search spend at PRICE_WEB_SEARCH * max_uses
                    # per run. Fetches are billed only as input tokens.
                    {"type": "web_search_20260209", "name": "web_search", "max_uses": 130},
                    {"type": "web_fetch_20260209", "name": "web_fetch"},
                ],
                "messages": messages,
            }
            # Resume in the same container across pause_turn continuations.
            if container_id is not None:
                stream_kwargs["container"] = container_id

            with client.messages.stream(**stream_kwargs) as stream:
                message = stream.get_final_message()

            totals["api_calls"] += 1
            accumulate_usage(totals, message.usage)
            text_parts.extend(
                block.text for block in message.content if block.type == "text"
            )
            # Carry the container forward so the next continuation resumes the
            # pending code-execution tool uses instead of being rejected.
            container = getattr(message, "container", None)
            if container is not None:
                container_id = container.id

            if message.stop_reason != "pause_turn":
                break
            # Re-send the conversation with the paused assistant turn appended;
            # the API resumes the tool loop where it left off.
            messages.append({"role": "assistant", "content": message.content})

        return message, text_parts

    # Retry the whole scan if the stream dies mid-read. A dropped attempt's
    # partial output is unusable, but completed calls' usage is already in
    # `totals`, so the final log still reflects what the run actually cost.
    for attempt in range(1 + STREAM_RETRIES):
        try:
            message, text_parts = run_scan()
            break
        except (anthropic.APIConnectionError, httpx.TransportError) as exc:
            if attempt == STREAM_RETRIES:
                log_usage(totals)  # surface what the failed run still cost
                raise
            print(
                f"Stream dropped mid-run ({exc!r}); restarting scan "
                f"(retry {attempt + 1} of {STREAM_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(30 * (attempt + 1))

    log_usage(totals)

    if message.stop_reason == "pause_turn":
        raise ValueError(
            f"Run was still paused after {MAX_PAUSE_CONTINUATIONS} continuations; "
            "report is incomplete. Report not sent."
        )

    # If the model hit the output cap, the report is truncated mid-section.
    # Surface it instead of emailing a half-complete newsletter.
    if message.stop_reason == "max_tokens":
        raise ValueError(
            "Model response was truncated at the max_tokens limit; "
            "raise max_tokens. Report not sent."
        )

    full_text = "\n\n".join(text_parts)

    # During web search the model emits text blocks narrating each search before
    # producing the report. Drop everything before the first HTML tag so only the
    # report itself is emailed.
    match = re.search(
        r"<(?:!doctype|html|head|body|h[1-6]|div|p|ul|ol|table|section)\b",
        full_text,
        re.IGNORECASE,
    )
    if not match:
        # No HTML report was produced (e.g. the model only narrated, or the call
        # returned empty). Fail loudly rather than emailing raw search narration.
        raise ValueError("Model response contained no HTML report; nothing to send.")

    # Sanitize before returning: web-search content is untrusted and the model's
    # output is not a trusted source of safe HTML (see sanitize_html above).
    report = sanitize_html(full_text[match.start():])
    if not report.strip():
        raise ValueError("Report was empty after sanitization; nothing to send.")
    return report

# Dark, flat "sage" theme matching the shared design system. Five-color palette:
# bg #0f120d, surface #1d231c, accent/sage #7d9b83, text #e6e4db, strong #ffffff,
# plus the one sanctioned semantic color, danger #cf8f83 (used only by the
# "Avoid" hiring flag). Every other shade here is a precomputed blend of those —
# no new hues.
#
# The palette is applied as literal hex, NOT as CSS custom properties: :root/
# var() are unsupported in Outlook (Word engine) and unreliable in Gmail, so the
# email keeps its robust two-layer approach — critical colors set inline on the
# wrapper (survive even where a client drops <style>) and the rest from this
# trusted <style> block (Apple Mail fully, Gmail web/app broadly). The design
# system's token/var() mechanism governs the web apps; here the same values are
# hardcoded so the rendered look still matches the reference (vibe-shelf). Flat
# only: no gradients, no shadows — separation comes from borders and surface
# contrast (bg vs surface), per the design rules.
#
# Fonts: Space Grotesk (display: masthead, headings, tier/label lines) and Inter
# (body and all UI) are named in the font stacks with system fallbacks. The email
# makes NO external font request — the whole newsletter avoids outbound calls
# from the message (no beacons/leaks), and clients widely strip webfonts anyway —
# so the typefaces render where a client already has them and fall back cleanly
# otherwise.
EMAIL_STYLE = """
  :root { color-scheme: dark; supported-color-schemes: dark; }
  body { margin: 0; padding: 0; background: #0f120d; -webkit-text-size-adjust: 100%; }
  .wrap { background: #0f120d; padding: 24px 12px; }
  .email {
    max-width: 680px; margin: 0 auto;
    background: #0f120d; border: 1px solid #303b31; border-radius: 12px;
    padding: 4px 26px 14px;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #e6e4db; line-height: 1.55; font-size: 15px;
  }
  .masthead { padding: 22px 0 14px; border-bottom: 2px solid #7d9b83; margin-bottom: 8px; }
  .masthead .title { font-family: "Space Grotesk", "Inter", system-ui, sans-serif; font-size: 20px; font-weight: 700; color: #ffffff; letter-spacing: -0.01em; }
  .masthead .title .accent { color: #7d9b83; }
  .masthead .date { font-size: 12px; color: #909089; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
  .email h2 {
    font-family: "Space Grotesk", "Inter", system-ui, sans-serif;
    font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;
    color: #7d9b83; border-left: 4px solid #7d9b83; padding: 7px 0 7px 12px;
    margin: 34px 0 14px;
  }
  .email h3 { font-family: "Space Grotesk", "Inter", system-ui, sans-serif; font-size: 16px; font-weight: 600; color: #ffffff; margin: 0 0 7px; }
  .email h3.tier {
    font-family: "Space Grotesk", "Inter", system-ui, sans-serif;
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em; color: #7d9b83;
    margin: 22px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #303b31;
  }
  .email p { margin: 7px 0; }
  .email strong { color: #aeaea6; font-weight: 600; }
  .email a { color: #7d9b83; text-decoration: none; border-bottom: 1px solid #415143; }
  .email ul, .email ol { margin: 7px 0; padding-left: 20px; }
  .email li { margin: 4px 0; }
  .item {
    background: #1d231c; border: 1px solid #303b31; border-radius: 8px;
    padding: 14px 16px; margin: 0 0 14px;
  }
  .highlights {
    background: #1d231c;
    border: 1px solid #7d9b83; border-radius: 12px; padding: 16px 18px; margin: 16px 0 24px;
  }
  .highlights h3 { color: #ffffff; }
  .pill {
    display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: 0.02em;
    padding: 2px 10px; border-radius: 999px; border: 1px solid; line-height: 1.5;
    font-family: "Inter", system-ui, sans-serif;
    background: #1d231c;
  }
  /* Signal type is a neutral category tag, not a warning: it folds into the sage
     accent (quiet, outlined). */
  .pill-signal { color: #cdd8cf; border-color: #415143; background: #2c362c; }
  /* Hiring-window ramp, palette-only and ordered by desirability so the color
     still reads at a glance without introducing new hues: Hot = strongest
     positive (solid sage), Warm = quiet sage, Cold = neutral surface, and Avoid
     is the sole use of the semantic danger color. */
  .temp-hot { color: #0f120d; border-color: #7d9b83; background: #7d9b83; }
  .temp-warm { color: #cdd8cf; border-color: #415143; background: #2c362c; }
  .temp-cold { color: #909089; border-color: #303b31; background: #1d231c; }
  .temp-avoid { color: #cf8f83; border-color: #cf8f83; background: #1d231c; }
  .footer { margin-top: 26px; padding-top: 14px; border-top: 1px solid #303b31; color: #909089; font-size: 12px; }
"""


def build_html_email(report_fragment, date_str):
    """Wrap the sanitized report body in the trusted, dark-themed email shell."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#0f120d">
<style>{EMAIL_STYLE}</style>
</head>
<body style="background:#0f120d;color:#e6e4db;">
<div class="wrap" style="background:#0f120d;">
<div class="email" style="background:#0f120d;color:#e6e4db;">
<div class="masthead">
<div class="title">Weekly Tech Intel <span class="accent">Newsfeed</span></div>
<div class="date">{date_str}</div>
</div>
{report_fragment}
<div class="footer">Generated automatically from live web search. Verify every role and source before acting.</div>
</div>
</div>
</body>
</html>"""


def send_email(body):
    sender = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = sender
    msg["Subject"] = f"Weekly Tech Intel Newsfeed — {datetime.now(timezone.utc).strftime('%B %d, %Y')}"
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, app_password)
        server.sendmail(sender, sender, msg.as_string())


# A generation run is expensive, so a transient SMTP failure shouldn't silently
# lose it. We retry the send rather than archiving the report anywhere, because
# this repo is public and the report renders the CANDIDATE_PROFILE secret.
SEND_MAX_ATTEMPTS = 3
# Delay before each retry, indexed by the attempt that just failed. With
# SEND_MAX_ATTEMPTS == 3 only the first two rungs (30s, 120s) are reached; 300s
# is the next rung if the attempt count is ever raised.
SEND_BACKOFF_SECONDS = (30, 120, 300)


def send_with_retry(body):
    """Send the report with bounded, backed-off retries on transient failures.

    Retries on transient network/SMTP errors only. Logs the attempt number and
    the exception's class name — never the exception message (it can echo the
    recipient address or server response), the recipient address, or any part
    of the report body, because workflow logs on a public repo are public. On
    final failure raise a generic, content-free error and accept the lost run.
    """
    for attempt in range(1, SEND_MAX_ATTEMPTS + 1):
        try:
            send_email(body)
            return
        except (smtplib.SMTPException, OSError, TimeoutError) as exc:
            if attempt == SEND_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Send failed after {SEND_MAX_ATTEMPTS} attempts; "
                    "report discarded."
                ) from None
            delay = SEND_BACKOFF_SECONDS[attempt - 1]
            print(
                f"Send attempt {attempt} of {SEND_MAX_ATTEMPTS} failed "
                f"({type(exc).__name__}); retrying in {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)


if __name__ == "__main__":
    try:
        report_fragment = get_newsfeed()
        date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
        newsfeed = build_html_email(report_fragment, date_str)
        # INVARIANT: this repo is public. Report content must never reach any
        # publicly readable surface — no uploaded run outputs, no workflow logs,
        # no committed files. A paid run is protected by retrying the send, not
        # by writing the report anywhere durable. Do not add persistence here.
        send_with_retry(newsfeed)
    except Exception as exc:
        # Exit non-zero so the GitHub Action surfaces the failure instead of
        # reporting a green run after a bad or missing send.
        print(f"Newsfeed run failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Sent successfully.")
