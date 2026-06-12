# Weekly Tech Intel Newsfeed

A GitHub Actions pipeline that emails a weekly, dark-themed HTML intelligence
briefing for an executive job search in the GRC / Technology Risk space. Every
Monday it asks Claude (with live web search and page fetching) to scan for:

- **Open roles** (Director → VP/Chief level GRC, Tech Risk, AI Governance) —
  every posting is fetched and verified live before it's reported, tiered into
  "Strong fits" vs "Broader — worth a look", with stated compensation captured
- **Regulatory actions** (OCC, CFPB, SEC, FTC, FDA, FinCEN, EU…) — enforcement
  is treated as a hiring signal: companies that just got slapped are building
- **Org signals** — IPOs, funding, layoffs, charters, M&A, each rated with a
  hiring-window temperature (Hot / Warm / Cold / Avoid)
- **Content opportunities** — industry reports, incidents, AI governance, and
  GRC methodology developments worth a LinkedIn post, ending with one
  recommended post angle for the week

The result is sanitized, wrapped in a trusted email shell, archived as a
workflow artifact, and emailed via Gmail.

## Setup

1. Fork or clone the repo.
2. Add repository secrets (Settings → Secrets and variables → Actions):

   | Secret | Required | Purpose |
   |---|---|---|
   | `ANTHROPIC_API_KEY` | yes | Claude API access (web search + web fetch enabled) |
   | `GMAIL_ADDRESS` | yes | Gmail account that sends (and receives) the report |
   | `GMAIL_APP_PASSWORD` | yes | A Gmail [app password](https://myaccount.google.com/apppasswords), not your real password |
   | `CANDIDATE_PROFILE` | no | The real candidate profile (see below) |

3. The schedule is Mondays 15:00 UTC (`.github/workflows/weekly-newsfeed.yml`);
   trigger a test run any time via the workflow's "Run workflow" button.

### The candidate profile

The scan is personalized by a profile block injected into the prompt at run
time. The repo ships only a generic example (`DEFAULT_PROFILE` in
`scripts/newsfeed.py`); the real profile lives in the `CANDIDATE_PROFILE`
secret so personal job-search details — background, target companies, home
metro, preferences — never appear in the repository or its history.

A custom profile must keep the same shape as the example: a
`WHO THE CANDIDATE IS` section, then `---`, then a `WATCHLIST COMPANIES`
section ending in the company list, because the surrounding prompt refers back
to both (including "the candidate's home metro area" for location tiering).

## Design notes

- **Prompt-injection defense**: the report is built from live web content,
  which is untrusted. Model output is never trusted as safe HTML — it is run
  through an allowlist sanitizer (`nh3`) that strips images, scripts, styles,
  event handlers, and non-http(s)/mailto URLs before anything is emailed, so a
  malicious page can't plant tracking beacons or script in the email.
- **Verified-live job postings**: search snippets routinely surface dead
  roles, so the prompt requires every posting to be fetched and confirmed live
  (real detail page, active apply control) before it can appear.
- **Cost controls**: prompt caching, a hard cap on web searches per run, and
  per-run token/cost logging in the Action output.
- **Failure handling**: streaming (avoids the 10-minute non-streaming
  timeout), `pause_turn` continuation, full-scan retries on dropped streams,
  truncation detection, and the styled report is saved as an artifact before
  the SMTP send so a mail failure never loses a paid run.
