# Deploying the API on Hugging Face Spaces (no card required)

Hugging Face Spaces runs Docker containers on **free CPU** — no payment method
of any kind. It's the fallback after Google Cloud billing verification blocked
this account (closed twice: once auto-closed, once a rejected UPI attempt).

A **Space is its own separate git repository** — not your GitHub repo — so we
push a *copy* of just what the container needs (Dockerfile, `src/`, the small
processed datasets) into a new Space repo, alongside a Space-specific
`README.md` that tells HF how to build it. Your GitHub repo is untouched.

## 1. Create the Space (browser, ~2 min)

1. Sign up / log in at <https://huggingface.co> (free, email only — no card).
2. Click your profile → **New Space**.
3. Fill in:
   - **Space name**: `f1-strategy-api` (or anything)
   - **License**: MIT
   - **Select the Space SDK**: **Docker** → then **Blank** template
   - **Space hardware**: **CPU basic · Free**
   - Visibility: Public
4. Click **Create Space**. It'll show you a git URL like
   `https://huggingface.co/spaces/<your-username>/f1-strategy-api` — note it.

## 2. Push the API into it (Cloud Shell, or any terminal with git)

You can reuse the same Google Cloud Shell session from before (or any
terminal) — this step doesn't need GCP, just git.

```bash
# Get an access token first: huggingface.co -> profile icon -> Settings ->
# Access Tokens -> New token (role: write). Copy it -- you'll paste it as the
# password when git asks, in the next few commands.

git clone https://github.com/ShivekRanjan/f1-strategy-engine
cd f1-strategy-engine

# Stage a slim copy with only what the container needs.
mkdir /tmp/space && cd /tmp/space
cp -r ~/f1-strategy-engine/src .
cp ~/f1-strategy-engine/pyproject.toml .
cp ~/f1-strategy-engine/Dockerfile .
mkdir -p data/processed
cp ~/f1-strategy-engine/data/processed/dry_laps.parquet \
   ~/f1-strategy-engine/data/processed/track_status.parquet \
   ~/f1-strategy-engine/data/processed/race_laps.parquet \
   ~/f1-strategy-engine/data/processed/results.parquet \
   ~/f1-strategy-engine/data/processed/lstm_nextlap.npz \
   data/processed/
cp ~/f1-strategy-engine/deploy/huggingface_space/README.md .

git init
git add -A
git commit -m "Deploy F1 Strategy Engine API"
git branch -M main

# Replace <your-username> below with your actual HF username.
git remote add space https://huggingface.co/spaces/<your-username>/f1-strategy-api
git push --force space main
```

When `git push` asks for a username/password:
- **Username**: your Hugging Face username
- **Password**: paste the **access token** you created above (not your HF
  account password — HF, like GitHub, requires a token for git operations)

## 3. Watch it build

Back in the browser, your Space page shows build logs automatically (Docker
image build → container start). First build takes a few minutes. Once it says
**Running**, your API is live at:

```
https://<your-username>-f1-strategy-api.hf.space
```

Check `<that-url>/health` returns `{"status":"ok"}`.

## 4. Point the frontend at it

Vercel → your project → **Settings → Environment Variables** → change
`VITE_API_BASE` to that URL → **Redeploy**.

## 5. Keep it warm

Same GitHub Action as before follows any host via a repo variable — no code
change needed. Repo → **Settings → Secrets and variables → Actions →
Variables** → set `API_URL` to the new `hf.space` URL.

> HF Spaces on the free CPU tier also sleep after inactivity and wake in a few
> seconds (faster than Render's free tier); the keep-alive pinger avoids that
> almost entirely, same as before.

## Updating later

Free-tier Spaces don't auto-deploy from your GitHub repo. To ship a change,
redo step 2's copy + `git push --force space main` (or set up a small script —
ask if you want one).

## CORS (optional)

Add an environment variable in the Space's **Settings → Variables and
secrets**: `F1SE_CORS_ORIGINS` = `https://f1-strategy-engine.vercel.app` to
lock CORS to your frontend origin.
