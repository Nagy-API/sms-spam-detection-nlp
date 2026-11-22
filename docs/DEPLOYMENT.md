# Deployment (Vercel)

## What gets deployed

One Python 3.12 serverless function built from `app/main.py` (FastAPI preset), serving the API and
the static frontend from the same origin. `vercel.json` includes `models/spam-v2/**`, `models/trace-v3/**` and
`frontend/**` and excludes data, notebook, reports, docs, configs, tests, NLTK data, the legacy
pickles, and `spam_detector/training/`. `.vercelignore` keeps those out of the upload as well.
**No custom build command or model environment variables are required.** In Vercel Production,
the service always loads the bundled V2 and V3 model pair. Research directory overrides
(`SPAM_MODEL_DIR` and `TRACE_EXPERIMENTAL_MODEL_DIR`) do not apply there.

## Frozen Production behavior

The default TRACE screen uses `POST /predict/v3`. The bundled head may return **Potential
Phishing** as an educational experimental verdict. `GET /model/v3` must return
`phishing_status: "experimental"` and `independent_phishing_validation: false`. Briefs
describing a Production-disabled head refer to the earlier local-only prototype.

**Use spam-only mode** links to `/?model=v2` and uses the historical classifier at
`POST /predict`. It remains available if the V3 head cannot load or fails its baseline
compatibility check. Both modes explain that a Ham verdict does not guarantee safety.

Frozen model SHA-256 values:

| Model | Version | SHA-256 |
|---|---|---|
| Historical spam | `2.0.0` | `5f79aee6ba36e0f4e678010a1f5be95213a934cd29eb85bdb72d9eced526f34a` |
| Experimental phishing head | `3.0.0-rc.1` | `0f9118e7cc9f5007ad5ca06a624797b4e947da9cbb027a1ff966df23e4dff14a` |

## TRACE V3 release checklist

1. Reproduce the head with `python -m spam_detector.training.v3 export` and verify both artifact digests against the bundled metadata.
2. Run Ruff, the full Python suite and Playwright. Build a clean copy with Vercel CLI `vercel build --prod` using the existing project settings.
3. Wait for billing restoration, rerun mandatory GitHub checks, and resolve failures before consolidating the approved PRs in dependency order.
4. Confirm the exact main SHA and both model digests before staging the authorized release.
5. Stage Production without assigning its domain. Check `/health`, `/ready`, both model
   endpoints, all four V3 outcomes, the three V2 outcomes, the mode links, the experimental
   disclosure and the Ham safety caveat. Confirm both digests and exercise comparison on
   mobile and desktop.
6. Promote the verified staged deployment. Recheck anonymous public access, logs, provenance and rollback.

Phishing remains experimental unless the independent evaluation gates pass. Demo deployment must preserve that disclosure.

**Local RC build evidence:** Vercel CLI 59.23.2 completed `vercel build --prod` for the linked
project. Its function map specifies Python 3.12, Linux x86-64 dependencies and both verified
model assets. No raw corpora or environment files were mapped. Use a clean staging directory
without spaces on Windows, because the CLI can fail to quote the `uv` executable path.
Do not publish the local build or its downloaded project settings.

## Verified locally (this branch)

`npx vercel@59.23.2 build --target preview`, run on a clean `git archive` of the branch with
project settings from `vercel pull`, produced one function:

| | Phase 1 (v1) | This branch (v2) |
|---|---|---|
| Function bundle | 207.15 MiB | **20.05 MiB** (Linux wheels: 5.8 MiB vendored pip, 4.7 MiB pydantic-core, 2.9 MiB model) |
| Build step | download + verify NLTK data, strip NumPy/SciPy/sklearn test folders | none |
| Model load at cold start | joblib + NLTK lazy loaders | 0.05–0.19 s (JSON + checksum) |
| scikit-learn / NumPy / SciPy / NLTK in bundle | yes | **no** |

CI (`.github/workflows/ci.yml`) additionally installs **only** `requirements.txt` on Linux, starts
uvicorn, and checks real predictions. This proves the slim runtime is complete.

The local Vercel builder needs `uv` on `PATH` (`python -m pip install uv` in any environment).

## Preview deployment runbook

Unchanged in substance from Phase 1, and still the required procedure. Use the existing project
`nagy-apis-projects/sms-spam-detection-nlp` (`prj_di0asm5YAbI4JjCrvYS46lE3hYLd`). Stop if the
team or project differs, the project has no prior completed deployment, or Preview deployment
protection is disabled.

```powershell
git status --short --branch
git rev-parse HEAD
npx vercel@59.23.2 whoami
npx vercel@59.23.2 deploy --target=preview --scope nagy-apis-projects --project prj_di0asm5YAbI4JjCrvYS46lE3hYLd --no-wait --non-interactive
npx vercel@59.23.2 inspect DEPLOYMENT_ID --scope nagy-apis-projects --json
```

Confirm that the dashboard says **Preview** and that the metadata records the expected commit
before waiting for Ready. If Vercel reports Production or the environment cannot be confirmed,
cancel the build and report the deployment ID. The **Auto-assign Custom Production Domains**
setting can be disabled temporarily as an extra safeguard. Restore it afterwards and check for
unexpected Production deployments or aliases.

### Hosted checks for v2

While signed in to Vercel on the protected URL:

1. `GET /health` → `{"status":"ok"}`. `GET /ready` → `model_sha256` = `5f79aee6…f34a`.
2. `POST /predict` for ham, spam, each inconclusive reason, and 5,000 emoji code points. Compare
   with `tests/golden/golden_predictions.json`.
3. 422 for blank and oversized text, with no `input` echoed. 413 for a > 64 KiB body.
4. Run the browser flows from `tests/browser` against the URL.
5. Runtime logs: one `model_loaded` line per instance with `load_ms`, JSON `request` lines **without
   message text**, and no 5xx.
6. Record first-request and repeat-request timings to compare with Phase 1 (6.7 s first `/health`).

## Hosted verification record (2026-11-21)

Two protected Previews of the same model artifact were verified. Both were classified as Preview by
the deployment API (`target: null`, `readySubstate: STAGED`) and by `vercel inspect`. Anonymous
requests got a 302 to Vercel SSO. The project has only a `preview` target, no aliases, and zero
Production deployments.

| | Acceptance run | Fix re-verification |
|---|---|---|
| Deployment | `dpl_2muFmc4eoxsYf5ab2MAEmFVrCS2h` (`…-cy2ze51cb-…`) | `dpl_yue784Q8t1U57sVGsUqkdqnmEASC` (`…-lioe67vem-…`) |
| Commit | `cc8557247cd4f252d07bd2571b067f039d242673` | `f42de90955ff5a2a71389ec1ef662ac1bc88255f` |
| `/ready` model | v2.0.0, `5f79aee6…f34a` | v2.0.0, `5f79aee6…f34a` |
| Scripted API check groups¹ | 24/25 (`GET /predict` → 404, fixed in `f42de90`) | **25/25** |
| Playwright on hosted URL | 19/19 | **19/19** |
| Cold starts in logs | 2, `load_ms` 77–79 | 3, `load_ms` 69–74, peak RSS 63 MiB |
| Errors / 5xx / message text in logs | 0 / 0 / none | 0 / 0 / none |

¹ Groups include 16 golden predictions (spam, ham, every abstention reason, Unicode), 8 sanitized
422 inputs, 413 (including a chunked body), 405, the boundary cases at 5,000 code points, byte-identical
static assets, all security headers, and the CSP.

Latency: the first request after deployment took 2.0 s end-to-end (Phase 1: 6.7 s). Server-side
`/predict` ran at p50 2.3 ms and p95 9.0 ms. The client's warm round trip was 0.35–0.45 s from
the test machine to `iad1`, which is network-dominated. Peak process memory is 63–64 MiB,
including the Python runtime and the loaded model.

### Running the checks against a protected Preview

`npx vercel@59.23.2 curl <url>/health --scope nagy-apis-projects` authenticates through Deployment
Protection. **On first use it creates a project-level "Protection Bypass for Automation" secret**
(Settings → Deployment Protection). Anyone holding that secret can bypass protection, so revoke it
after testing if it is not needed. To run the browser suite against a Preview:

```powershell
$env:PLAYWRIGHT_BASE_URL = "https://<preview-host>"
$env:VERCEL_AUTOMATION_BYPASS_SECRET = "<secret from the project settings>"
npm --prefix tests/browser test
```

## Rollback / recovery

* The Phase 1 Preview (`dpl_AUNN1zDzoCR3m8DVjpyvev9kgUnT`, commit `616665d`) is untouched by this
  branch and remains the validated fallback.
* A v2 Preview is a new immutable deployment. If it misbehaves, keep using or re-promote the
  previous Preview. No data migration is involved.
* Production is not deployed by this branch. Promoting anything to Production needs explicit
  owner approval.
