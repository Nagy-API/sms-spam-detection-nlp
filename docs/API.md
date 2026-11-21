# API contract and frontend integration guide

The service is a single FastAPI app (`app/main.py`) that serves the static frontend and the API
from the **same origin**, so the browser needs no CORS and no API URL configuration. Interactive
schema: `/docs` (Swagger UI) and `/openapi.json`.

## `POST /predict`

Request (`application/json`):

```json
{ "text": "Congratulations! You won a free prize. Call now to claim." }
```

| Field | Rules |
|---|---|
| `text` | string, **1–5,000 Unicode code points** (an emoji counts as one; JavaScript must count with `[...text].length`, not `.length`), not only whitespace (Python `str.isspace` semantics, which includes U+001C–U+001F) |

Request bodies over **64 KiB** are rejected with 413 before parsing (the 5,000-code-point limit
fits even when every character is JSON-escaped as a surrogate pair).

### 200: classified

```json
{ "label": "spam", "spam_probability": 0.9648, "reason": null, "model_version": "2.0.0" }
```

### 200: inconclusive (the model abstains)

```json
{ "label": "inconclusive", "spam_probability": null, "reason": "unsupported_script", "model_version": "2.0.0" }
```

| Field | Type | Meaning |
|---|---|---|
| `label` | `"spam"` \| `"ham"` \| `"inconclusive"` | `spam` when calibrated P(spam) ≥ the model's threshold (0.75) |
| `spam_probability` | number 0–1 (4 decimals) \| `null` | Calibrated estimate on data like the training corpus. `null` iff inconclusive |
| `reason` | `null` \| `"no_letters"` \| `"unsupported_script"` \| `"no_recognized_features"` | Why it abstained. **Treat unknown values as a generic inconclusive** (new reasons may be added) |
| `model_version` | string | Semantic version of the served model |

Clients must ignore unknown response fields (new fields may be added in minor versions).

### Errors

| Status | Body | When | Suggested UI message |
|---|---|---|---|
| 422 | `{"detail":[{"type","loc","msg"}]}` | Missing/blank/non-string `text`, `string_too_long`, invalid JSON | `string_too_long` → length message. `string_too_short`, `value_error`, `missing` → "enter a message". Otherwise → "check your input" |
| 413 | `{"detail":{"type":"payload_too_large","msg"}}` | Body > 64 KiB | Length message |
| 503 | `{"detail":{"type":"model_unavailable","msg"}}` | Artifact missing, corrupted, or incompatible | "Service unavailable, try later" |
| 500 | `{"detail":{"type":"internal_error","msg"}}` | Unexpected bug | Generic error |

422 bodies never echo the submitted text (FastAPI's default does; we override it).

## `GET /health`, `GET /ready`, `GET /model`

| Endpoint | 200 body | Purpose |
|---|---|---|
| `/health` | `{"status":"ok"}` | Liveness. Does not touch the model |
| `/ready` | `{"status":"ready","model_version","model_sha256"}` (503 if not) | Readiness. A verified model is loaded |
| `/model` | `model_version`, `model_sha256`, `estimator`, `labels`, `decision_threshold`, `max_message_chars`, `language`, `intended_use`, `evaluation` (CV and locked-test summary with caveat) | Lets the UI describe the model from data instead of hard-coding claims |

## Headers

Every response carries `X-Request-ID` (echoes an incoming `X-Request-ID`, otherwise generated;
include it in bug reports), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and
`Referrer-Policy: no-referrer`. API responses add `Cache-Control: no-store`. Frontend pages get a
strict CSP: `script-src 'self'; style-src 'self'; connect-src 'self'`, and no inline scripts or styles.
**A redesign that adds inline `<script>`/`<style>`, web fonts, or third-party origins must update
`_FRONTEND_CSP` in `app/main.py`.**

## Frontend redesign handoff

What the current UI does and the redesign must keep (each item is covered by
`tests/browser/prediction.spec.cjs` or `tests/integration/test_api.py`):

1. **Input rules mirror the API:** count code points, block blank/whitespace-only and > 5,000
   before sending, and accept `.txt` uploads ≤ 5,000 code points without replacing current text on
   rejection.
2. **Stale results never show:** editing, uploading, or clearing cancels in-flight requests and hides
   the previous result.
3. **Three outcomes plus reason text:**

   | label / reason | Title | Description used today |
   |---|---|---|
   | `spam` | Likely Spam | patterns the model associates with SMS spam |
   | `ham` | Likely Ham | patterns associated with legitimate SMS |
   | `inconclusive` / `no_letters` | Inconclusive | no words to analyze |
   | `inconclusive` / `unsupported_script` | Inconclusive | only English SMS in the Latin alphabet is supported |
   | `inconclusive` / `no_recognized_features` or unknown | Inconclusive | no words or phrases the model recognizes |

4. **No safety claims.** Do not say "safe", "verified", "malicious", or "high accuracy", and do not
   present "ham" as a guarantee. Phase 1 tests assert this.
5. **Probability is available but not displayed today.** If the redesign shows it, call it an
   estimate ("model estimate: 96% likely spam"), not "confidence", and link to `/model` evaluation
   caveats. Show nothing for `null`. Product sign-off is needed before showing it.
6. **Accessibility:** labeled textbox with description, `role="alert"` for errors,
   `role="status"` for progress/results, and visible focus, with no horizontal overflow down to 320 px.
7. **Same-origin fetch** to `/predict`. Keep the `const API_URL = "/predict"` line or update the
   test that parses it.

The redesign needs no backend or ML change. Model updates keep this contract and change only
`model_version`.
