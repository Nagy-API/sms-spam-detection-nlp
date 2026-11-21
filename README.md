# SMS Spam Detection NLP

ShieldSMS classifies English SMS messages as likely spam or likely ham (legitimate). The web interface accepts pasted messages and `.txt` files. It reports an inconclusive result when the model finds no recognized words or phrases.

## Project structure

| Path | Purpose |
| --- | --- |
| `app/main.py` | FastAPI service. It cleans and tokenizes text, removes English stop words, and lemmatizes tokens using NLTK. A unigram and bigram `CountVectorizer` feeds a multinomial Naive Bayes classifier. |
| `frontend/` | Static HTML, CSS, and JavaScript that calls the same-origin API. |
| `models/` | Project-generated scikit-learn classifier and vectorizer, saved with joblib. |
| `notebook/sms_spam_detection.ipynb` | Data exploration, preprocessing, training experiments, and evaluation on the SMS dataset. |
| `requirements.txt` | Python dependencies for the API. |

## Install and run locally

Use Python 3.12. From the repository root, create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with `source .venv/bin/activate`. Install the pinned dependencies and prepare the pinned NLTK resources once before starting the app. The saved artifacts were serialized with scikit-learn 1.8.0.

```sh
python -m pip install -r requirements.txt
python scripts/prepare_nltk.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the [local interface](http://127.0.0.1:8000/). The same server provides `GET /health` and accepts `POST /predict` with a JSON `text` field. The UI and API accept up to 5,000 Unicode code points, counting each emoji code point once. The server loads the included joblib artifacts and local NLTK resources at startup. A missing artifact stops startup. [Interactive API documentation](http://127.0.0.1:8000/docs)

| HTTP status | Response |
| --- | --- |
| `200` | `label` is `ham` or `spam` and `spam_probability` is a number available for evaluation. The browser displays the label. |
| `200` | `{"label":"inconclusive","spam_probability":null}` when the processed text has no recognized vocabulary features. |
| `422` | FastAPI validation details for blank or whitespace-only text, text over the limit, a missing field, or an invalid type. |

## Tests

After preparing the NLTK resources, run these commands from the repository root.

```sh
python -m pip install "httpx2==2.16.2"
python -m unittest discover -s tests -v
```

With the Python environment still active, run the browser regressions from the repository root. Playwright starts the local FastAPI server on port 8765 and runs the suite in Chromium.

```sh
npm --prefix tests/browser ci
npm --prefix tests/browser exec -- playwright install chromium
npm --prefix tests/browser test
```

## Vercel preview

Use the existing `nagy-apis-projects/sms-spam-detection-nlp` project (`prj_di0asm5YAbI4JjCrvYS46lE3hYLd`). Vercel can assign the first deployment of a new project to Production even when the CLI requests Preview. Stop if the team or project differs, the project has no prior completed deployment, or Preview deployment protection is disabled. Provisioning a new project requires a separate decision. [Vercel first-deployment guidance](https://github.com/vercel/vercel-plugin/blob/main/commands/deploy.md)

From the repository root, check the branch, commit and working tree. Confirm that the authenticated CLI account has access to the team, then open the exact project in the dashboard and verify that a previous protected Preview is Ready. The project uses the FastAPI framework preset and requires no application environment variables.

```sh
git status --short --branch
git rev-parse HEAD
npx vercel@59.23.2 whoami
npx vercel@59.23.2 teams ls
```

The **Auto-assign Custom Production Domains** setting can be temporarily disabled as an extra safeguard against assigning Production aliases. It controls alias assignment only. Phase 1 also tested a `--local-config` build guard, but Vercel used the build command committed in `vercel.json`. If a build-time environment guard is required, review and commit it in the effective project configuration before uploading. [Vercel deployment controls](https://vercel.com/docs/deployments/managing-deployments) · [Vercel project configuration](https://vercel.com/docs/project-configuration/vercel-json)

Deploy with the target, scope and project pinned. Replace `DEPLOYMENT_ID` with the returned ID and inspect it immediately. Confirm the dashboard says **Preview** and the deployment metadata records the expected Git commit before waiting for Ready. A `null` target in the deployment API denotes standard Preview. If Vercel reports **Production** or the environment cannot be confirmed, stop, cancel any ongoing build in the dashboard and report the deployment ID. After a confirmed Preview, restore the original domain setting if it was changed and check for unexpected Production deployments or aliases. [Vercel CLI deploy](https://vercel.com/docs/cli/deploy) · [Vercel Preview environments](https://vercel.com/docs/deployments/environments#preview-environment-pre-production)

```sh
npx vercel@59.23.2 deploy --target=preview --scope nagy-apis-projects --project prj_di0asm5YAbI4JjCrvYS46lE3hYLd --no-wait --non-interactive
npx vercel@59.23.2 inspect DEPLOYMENT_ID --scope nagy-apis-projects --json
```

Open the unique deployment URL while signed in to Vercel. Check `/health`, the frontend assets, and real ham, spam and inconclusive predictions from `/`. Confirm that the browser uses the same-origin `POST /predict` and that the API returns `422` for blank and oversized text. Use the deployment's Build Logs and Runtime Logs to check function startup and record first and repeat request timings. [Vercel Authentication](https://vercel.com/docs/deployment-protection/methods-to-protect-deployments/vercel-authentication)

Vercel builds one Python 3.12 function from `app/main.py` and serves the frontend from the same origin. The build downloads and verifies four upstream NLTK archives and bundles the required resources, including the English Punkt tables, alongside the two existing model files. The build strips NumPy, SciPy, and scikit-learn's upstream test directories from its disposable build environment to fit the function bundle. The deployment excludes the SMS training dataset, notebook, project tests, and offline training script. [Vercel FastAPI guide](https://vercel.com/docs/frameworks/backend/fastapi) · [Python runtime](https://vercel.com/docs/functions/runtimes/python)

## Dataset and retraining

The included model files let the API run without the dataset. For retraining or running the notebook, download `spam.csv` from the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it at `data/spam.csv`, creating `data/` if needed. The project expects the CSV's `v1` and `v2` columns and reads it as Latin-1. The exact CSV is kept local because the Kaggle listing does not identify a redistribution license and the messages include contact details.

The original SMS Spam Collection is attributed to Tiago Almeida and Jos Hidalgo (2011) and is distributed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). UCI provides the corpus in a different file format. The project reads the Kaggle CSV representation and transforms its SMS text into unigram and bigram counts for training. [UCI dataset citation](https://doi.org/10.24432/C5CC84)

To run the notebook, install the additional packages and start Jupyter from `notebook/`.

```sh
python -m pip install "pandas==3.0.0" jupyterlab matplotlib seaborn imbalanced-learn
cd notebook
jupyter lab sms_spam_detection.ipynb
```

To regenerate artifacts offline, use a separate checkout with the CSV and prepared NLTK resources. From its repository root, install `pandas==3.0.0` and run `python -m app.train`. That command writes both model files in the checkout where it runs.
