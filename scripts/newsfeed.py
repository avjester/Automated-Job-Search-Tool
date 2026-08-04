import anthropic
import httpx
import nh3
import re
import smtplib
import os
import sys
import time
from datetime import datetime, timezone
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
# (item cards, tier headers) that our trusted stylesheet targets. They are
# cosmetic only and don't widen the security surface: `class` cannot execute
# or exfiltrate, and the `style` attribute, `<style>` element, scripts,
# images, and event handlers all remain stripped. Worst case from an injected
# class is a misplaced tier header — a visual nuisance, not a vulnerability.
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
# that follows it refers back to both.
DEFAULT_PROFILE = """WHO THE CANDIDATE IS

The candidate has 10+ years of experience in Risk Management, spanning Operational Risk and Enterprise Risk (ERM) functions at large organizations. They are looking for Director, Senior Director, or VP level Risk Management, Operational Risk, or Enterprise Risk leadership roles. Target verticals are function-first rather than regulation-driven: financial services, consumer goods, technology, and travel/hospitality all fit the profile equally.

The candidate is remote-based in Chicago, IL; treat that as their home metro area. They are open to fully remote or hybrid roles.

---

WATCHLIST COMPANIES

American Express, Capital One, Cisco, Coca-Cola, Expedia."""


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
# budget — search spend is capped by max_uses on the web_search tool. Left at
# the original 9-category value: watchlist checks (up to 15 companies), an
# ATS-wide sweep across six job-board domains, aggregator checks, and a live
# web_fetch verification per candidate role can still add up to more tool
# iterations than the search cap alone suggests, so this guard keeps its
# margin even though the search budget below is much lower.
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

    profile = os.environ.get("CANDIDATE_PROFILE", "").strip() or DEFAULT_PROFILE

    prompt = f"""Today's date is {today}.

You must use live web search for every item in this report. Do not rely on your training data for any factual claim about a company or a role. If you cannot find a live, dated source for an item, do not include it. You also have a web_fetch tool: use it to open every candidate job posting directly and confirm from the fetched page — never from a search snippet — that the role is still live and open to applications.

You are a research assistant supporting a Risk Management executive — referred to throughout as "the candidate" — who is actively searching for a Senior, Director, or VP+ level role.

---

{profile}

This watchlist is a starting point, not a boundary. The search is profile-driven, not list-driven: any company in the candidate's target verticals is in scope regardless of whether it appears above. Expect most of the best findings each week to come from companies NOT on the watchlist.

---

OPEN ROLES SCAN

ROLE TITLES IN SCOPE. Search for Senior, Director, Senior Director, VP, SVP, and Head of level roles across the full family of titles this function goes by, not just the literal string "Risk Management": Risk Management, Operational Risk, Enterprise Risk (ERM), Non-Financial Risk, Business Risk, Operational Resilience, and second-line Risk Oversight. At startups and growth-stage companies, "Head of Risk" or "Head of X Risk" is typically Director-to-VP equivalent — treat it as in scope.

DISCOVERY STRATEGY. The search is profile-driven: most qualifying roles each week will be at companies not on the watchlist, so do not simply iterate the watchlist company by company. Run these discovery passes, in this order:

1. Watchlist companies: check the careers pages of the watchlist companies above for openings matching the role titles in scope.
2. ATS-wide title sweeps: run site-restricted web searches for the role titles above directly across the major applicant-tracking-system domains — boards.greenhouse.io, jobs.lever.co, jobs.ashbyhq.com, myworkdayjobs.com, jobs.smartrecruiters.com, apply.workable.com — for example: site:boards.greenhouse.io "Director" "Operational Risk". This is the highest-yield way to find companies the candidate has never heard of. Filter the hits to companies matching the candidate's target verticals.
3. Job aggregators for discovery: LinkedIn Jobs, Built In (the candidate's home metro and remote), Wellfound, and Welcome to the Jungle/Otta. Aggregators are for discovery only — always follow through to the underlying company posting and cite that as the Source, never the aggregator page.
4. IPO pipeline: scan recent S-1 filings on SEC EDGAR and credible IPO-pipeline coverage for companies in the candidate's target verticals approaching public markets within 18 months — these are high-signal hiring windows where Risk Management investment is most active — and check those companies' careers pages.

Aim for breadth of companies over exhaustive depth on any one company. A weekly report that surfaces 8 to 15 verified roles across many companies is more useful than 3 roles from the watchlist plus an exhausted search budget.

PRESENT ROLES IN TWO TIERS. Search broadly, but do not present the results as one flat list — breadth is valuable for discovery but creates noise when every role is shown with equal weight. Split the confirmed roles into two labeled sub-sections, strongest first within each:

- "Strong fits": roles where title/function, seniority (Senior, Director, Senior Director, VP, SVP, or Head of level), AND vertical all clearly match the candidate's profile, and the location is the candidate's home metro area or fully remote. These are the roles they should look at first.
- "Broader — worth a look": real, verified, currently-live roles that are a stretch on one dimension — seniority slightly off, vertical adjacent rather than core, or location out-of-area with an unclear or onsite arrangement. Include these for discovery value, but cap this tier at the 8 strongest; if more than 8 qualify, keep the 8 best fits to the candidate's profile and drop the rest rather than padding the list.

Do not relax the liveness/verification rules for either tier — a role must be fetched and confirmed live to appear in either. Tiering is about ranking what you found, never about lowering the bar for what counts as verified. If a role is a genuine strong fit, it goes in "Strong fits" even if it is the only role this week.

There is no recency window on this scan — roles are governed by whether they are currently live, not by when they were first posted. A still-open role posted three weeks ago is in scope; a role posted yesterday that has already closed is not.

Every posting you include must be currently live and open to applications. Search results and search-engine snippets routinely surface roles that have already been filled or closed, so a search hit is not sufficient evidence that a role is open. Before including any role, open the posting page itself with web_fetch and confirm from the fetched content that it is still accepting applications. A role you did not fetch does not go in the report.

A page that returns successfully is NOT proof the role is live. Closed postings very frequently still "work" but silently redirect to the company's default careers homepage, a job-search index, or a generic "open positions" listing, while the original link continues to resolve. You must confirm that the final page you land on actually displays that exact role — its specific title and description, with an active apply control. If the link instead lands on a careers homepage, a job-search or "open positions" index, a search results page, or a "job not found" / "this position is no longer available" page, the role is dead — exclude it. The link you put in the Source field must point to that live, role-specific detail page, not to a redirect target or a careers landing page.

Reject the posting — do not list it — if any of the following are true: the page does not load or returns an error; the link redirects to or lands on a generic careers page, job-search index, or listing rather than the specific role's detail page; the final page does not display that exact role's title and description with an active apply control; the page states the role is closed, filled, paused, on hold, expired, or "no longer accepting applications"; the listing shows no posting or last-refreshed date; or the posting date is more than 30 days before today. When in doubt, exclude rather than guess: an omitted role is fine, a dead role is the failure mode to avoid.

For each confirmed role, provide the role title, company, the posting or last-refreshed date exactly as it appears on the page, and a direct link to the role-specific posting itself (not a search results page, careers homepage, or job-aggregator listing). While you have the posting open to verify it is live, also capture the stated compensation range if one is present — US postings frequently disclose it under pay-transparency laws — and report it exactly as written. If the company is in IPO preparation or publicly known to be approaching IPO within 18 months, flag this prominently — it is a high-priority hiring signal. If you cannot confirm a single live role this week, output: "Nothing confirmed this week."

For every confirmed role, note its location and work arrangement (remote, hybrid, or onsite) as stated on the posting. If the role's primary location is outside the candidate's home metro area, additionally flag the company's current work-location posture: whether it has recently announced or enforced a significant Return-to-Office (RTO) mandate, or whether it is genuinely remote-friendly. Base this on dated, verifiable sources — the posting's own remote/location terms, a company announcement, or recent news coverage — and say so briefly if you cannot confirm either way. This flag is informational only: do NOT exclude, downrank, or filter out an otherwise relevant out-of-area role because of an RTO push or because the work arrangement is unclear. The candidate still wants to see these roles; the flag simply tells them what they would be walking into. Roles based in the candidate's home metro area, or explicitly advertised as fully remote, do not need the RTO research.

---

SEARCH BUDGET

Spend the search budget on the discovery passes above, in the order listed — watchlist checks first, then the ATS-wide sweep (the highest-yield pass), then aggregators, then IPO-pipeline screening. Live-verification of individual postings is done with web_fetch, which does not draw from this search budget.

---

OUTPUT FORMAT

Begin your response with the opening HTML tag. Do not narrate your search process, describe your methodology, summarize what you are about to do, or include any preamble or transitional language before the HTML output. The report starts with the HTML — nothing before it.

At the top of the report, flag the three strongest roles overall (by fit to the candidate's profile, with named watchlist-company involvement breaking ties), wrapped in <div class="highlights">…</div>, using the same item fields as below.

For each role, provide:
- What happened: one to two sentences, factual and specific.
- Why it matters to the candidate: one to two sentences on how this fits the job search.
- Recommended action: a specific next step and, where relevant, a time window.
- Fit: one sentence stating why this role fits the candidate's profile and the single biggest caveat or stretch (e.g. "Core Enterprise Risk leadership at a growth-stage fintech; caveat: onsite NYC with no stated remote option"). This is what lets the candidate skim-accept or skim-reject in one read.
- Location and work arrangement: the role's location and whether it is remote, hybrid, or onsite. For roles outside the candidate's home metro area, also flag whether the company has a recent Return-to-Office (RTO) push or is remote-friendly, with the basis for that flag. This is informational and never a reason to omit the role.
- IPO status (if applicable): whether the company is in IPO preparation or approaching IPO within 18 months, with the basis (announced plans, S-1 filing, recent funding, public news).
- Compensation: the pay range exactly as stated on the posting you fetched, including what it covers (base, on-target earnings, bonus, equity) if specified — e.g. "$180K–$220K base + bonus." If the posting shows no range, write "Not disclosed on posting"; you may add a market estimate ONLY if you find a dated, citable public source and label it clearly as an estimate with that source. Never invent or guess a number from general knowledge.
- Source: direct link to the role-specific posting itself.

When flagging errors and limitations, apply the following rules throughout the report.
If a job posting cannot be confirmed as currently live and accepting applications by opening the posting page, exclude it entirely. Do not list unverified or stale roles even with a caveat — in this report a wrong listing is worse than an omission.
If web search returns no results for a specific watchlist company, do not infer absence of openings. Note it as: "No confirmed results found for [company] this week — coverage may be incomplete."

FORMAT AND MARKUP

Output ONLY the report body as an HTML fragment. Do NOT include <!doctype>, <html>, <head>, <body>, <style>, or any CSS — a styling shell is wrapped around your output automatically. Do not set any colors, fonts, or style attributes yourself; the only styling you control is the class names listed below, which hook into that shell. Do not invent other class names or use any class not listed here.

Structure:
- <h2> for the report's single section header (e.g. "OPEN ROLES").
- <h3> for item titles (the role title + company).
- Wrap every individual item in <div class="item">…</div>.
- <strong> for field labels (e.g. <strong>Why it matters to the candidate:</strong>).
- Plain prose in <p> tags; lists in <ul>/<li>. Use <a href="…"> for every source link.
- For the three highest-priority roles at the very top, wrap that whole block in <div class="highlights">…</div>.
- Render the two tier sub-headers as <h3 class="tier">Strong fits</h3> and <h3 class="tier">Broader — worth a look</h3>, each followed by that tier's item divs.

Do not use markdown. No inline JavaScript, no images, no tables. Keep nesting shallow and clean.
"""

    totals = {
        "input": 0, "cache_write": 0, "cache_read": 0, "output": 0,
        "searches": 0, "api_calls": 0,
    }

    def run_scan():
        """One full scan: stream the request, following pause_turn continuations.

        The server-side tool loop pauses (stop_reason "pause_turn") after ~10
        tool iterations. Keep continuing the same conversation until the model
        finishes for real. Streaming avoids the SDK's 10-minute non-streaming
        timeout.
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
                    # Sized for a single-category (roles-only) scan: up to 15
                    # watchlist-company checks, an ATS-wide sweep across 6
                    # job-board domains with a few title variants each,
                    # aggregator checks, and IPO-pipeline screening, plus
                    # headroom for query reformulation. Raise this if the
                    # per-run search count logged below is regularly hitting
                    # the cap.
                    {"type": "web_search_20260209", "name": "web_search", "max_uses": 60},
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
# bg #0f120d, surface #1d231c, accent/sage #7d9b83, text #e6e4db, strong #ffffff.
# Every other shade here is a precomputed blend of those — no new hues.
#
# The palette is applied as literal hex, NOT as CSS custom properties: :root/
# var() are unsupported in Outlook (Word engine) and unreliable in Gmail, so the
# email keeps its robust two-layer approach — critical colors set inline on the
# wrapper (survive even where a client drops <style>) and the rest from this
# trusted <style> block (Apple Mail fully, Gmail web/app broadly). Flat only:
# no gradients, no shadows — separation comes from borders and surface
# contrast (bg vs surface), per the design rules.
#
# Fonts: Space Grotesk (display: masthead, headings, tier/label lines) and Inter
# (body and all UI) are named in the font stacks with system fallbacks. The email
# makes NO external font request — the whole newsletter avoids outbound calls
# from the message (no beacons/leaks), and clients widely strip webfonts anyway —
# so the typefaces render where a client already has them and fall back cleanly
# otherwise.
#
# Note: the pill/tag styles (signal-type and hiring-window-temperature) from the
# original nine-category design were dropped along with those categories — this
# roles-only scan has no field that needs them, so the CSS stays free of dead
# selectors the model is never instructed to emit.
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
<div class="title">Weekly Open Roles <span class="accent">Scan</span></div>
<div class="date">{date_str}</div>
</div>
{report_fragment}
<div class="footer">Generated automatically from live web search. Verify every role and source before applying.</div>
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
    msg["Subject"] = f"Weekly Open Roles Scan — {datetime.now(timezone.utc).strftime('%B %d, %Y')}"
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

