# Deployment (Vercel)

## What gets deployed

One Python 3.12 serverless function built from `app/main.py` (FastAPI preset), serving the API and
the static frontend from the same origin. `vercel.json` includes `models/spam-v2/**` and
`frontend/**` and excludes data, notebook, reports, docs, configs, tests, NLTK data, the legacy
pickles, and `spam_detector/training/`. `.vercelignore` keeps those out of the upload as well.
**No build command and no environment variables are required.** (Optional: `SPAM_MODEL_DIR`
overrides the artifact directory.)

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

## Rollback / recovery

* The Phase 1 Preview (`dpl_AUNN1zDzoCR3m8DVjpyvev9kgUnT`, commit `616665d`) is untouched by this
  branch and remains the validated fallback.
* A v2 Preview is a new immutable deployment. If it misbehaves, keep using or re-promote the
  previous Preview. No data migration is involved.
* Production is not deployed by this branch. Promoting anything to Production needs explicit
  owner approval.
