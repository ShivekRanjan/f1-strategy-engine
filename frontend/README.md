# F1 Strategy Engine — frontend

A React + Vite + TypeScript + Tailwind client for the FastAPI engine. All
modelling lives in the Python package; this app is a pure API client.

## Develop

```bash
npm install
npm run dev        # http://localhost:5173
```

It reads the API base from `VITE_API_BASE` (see `.env.example`), defaulting to
`http://localhost:8000`. Start the backend first:

```bash
# from the repo root
uvicorn f1se.api:app --reload
```

## Build / check

```bash
npm run build      # tsc (strict) + vite production build -> dist/
npm run typecheck  # types only
```

## Stack

- **Vite + React 18 + TypeScript** (strict).
- **Tailwind** design tokens — the F1 broadcast theme lives in `tailwind.config.js`;
  reusable components in `src/components/`.
- **Recharts** for the distribution / championship / lap-pace charts.
- Typed API client in `src/api/`; one view per tab in `src/views/`.

## Deploy (Vercel or Netlify)

The frontend is a static SPA; deploy it separately from the API.

**Vercel** — import the repo, set:
- Root Directory: `frontend`
- Build Command: `npm run build` · Output Directory: `dist`
- Environment variable: `VITE_API_BASE = https://<your-api-host>` (the deployed
  FastAPI URL, e.g. on Render).

**Netlify** — same idea: base `frontend`, build `npm run build`, publish `dist`,
and set `VITE_API_BASE`.

Remember to set `F1SE_CORS_ORIGINS` on the API to your frontend's origin.
