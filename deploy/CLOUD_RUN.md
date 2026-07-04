# Deploying the API on Google Cloud Run

Cloud Run gives each request a **full vCPU** (vs Render free's ~0.1), so the
heavy Monte-Carlo searches run several times faster there — a 3-stop strategy
search drops from ~60–100 s on Render free to ~10 s. It has a generous
always-free tier (2M requests, 360k vCPU-seconds, 180k GiB-seconds per month),
which comfortably covers a portfolio app. A billing account (card on file) is
required even for the free tier; with `--max-instances 1` you cannot be
surprise-billed for a traffic spike.

The container is the **same `Dockerfile`** already used for Render — it honours
`$PORT` (Cloud Run injects `8080`), so nothing in the app changes.

## One-time deploy (no local install — use Cloud Shell)

1. Open <https://console.cloud.google.com>, **create a project**, and enable
   billing on it (Billing → link a billing account).
2. Click the **Cloud Shell** icon (`>_`, top-right). It's a free browser
   terminal with `gcloud`, `git`, and Docker pre-installed.
3. In Cloud Shell:

   ```bash
   git clone https://github.com/ShivekRanjan/f1-strategy-engine
   cd f1-strategy-engine

   gcloud run deploy f1se-api \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --cpu 1 \
     --min-instances 0 \
     --max-instances 1 \
     --cpu-boost \
     --timeout 120 \
     --set-env-vars F1SE_WARM_CACHES=1
   ```

   - First run prompts to enable the Run + Cloud Build APIs — say **yes**.
   - `--source .` builds the repo's `Dockerfile` via Cloud Build.
   - `--min-instances 0` = scale to zero (stay free when idle); `--max-instances 1`
     caps cost **and** keeps every request on one instance so the in-process
     recommend cache stays warm.
   - `us-central1` is an always-free-tier region.

   It prints a **Service URL** like `https://f1se-api-xxxxx-uc.a.run.app`.
   Check `<url>/health` returns `{"status":"ok"}`.

## Point the frontend at it

In Vercel → your project → **Settings → Environment Variables**, change
`VITE_API_BASE` to the Cloud Run URL (no trailing slash) and **redeploy**.

## Keep it warm + primed

The `Keep the API awake` GitHub Action already pings every 10 min. Point it at
the new host without editing code: repo **Settings → Secrets and variables →
Actions → Variables → New variable**, name `API_URL`, value the Cloud Run URL.

> On Cloud Run, CPU is throttled between requests, so the boot warm-up thread
> (`F1SE_WARM_CACHES`) makes only slow progress while idle — but that's fine:
> full per-request CPU means even a *cold* heavy request is ~10 s, the
> in-process caches persist while the pinger keeps the instance alive, and the
> pinger primes the Home-page caches directly. If you ever want everything
> instant and don't mind ~$5–15/mo, add `--no-cpu-throttling --min-instances 1`
> so CPU is always allocated and the warm-up completes at boot.

## Redeploying later

Every push to `main` does **not** auto-deploy to Cloud Run (unlike Render).
Re-run the same `gcloud run deploy …` command (or set up a Cloud Build trigger
on the repo) to ship a new version.

## Locking down CORS (optional)

The API defaults to permissive CORS (`*`) — fine for a public, read-only API.
To restrict it to your frontend origin, add
`--set-env-vars F1SE_CORS_ORIGINS=https://f1-strategy-engine.vercel.app` to the
deploy command (comma-separate multiple origins).
