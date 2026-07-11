# Deploying the website (public link, multi-user)

This gets your site onto a public URL that anyone can open, where each visitor
has their own private saved portfolios (no login required). Free hosting.

Everything is already prepared. You do the account/click steps — those can only
be done from your own accounts.

---

## What's already done for you

- **The site fits in free-tier memory.** `build_artifacts.py` precomputes a tiny
  2.8 MB data bundle (`website/data/`) so the app never loads the 2.7M-row dataset
  in production.
- **Multi-user isolation.** Each browser gets an anonymous session cookie; saved
  portfolios are private to that visitor. No accounts, no passwords, no personal data.
- **A database that survives restarts.** The app uses SQLite locally and
  PostgreSQL in production automatically (via the `DATABASE_URL` environment variable).
- **Production server + config.** `gunicorn`, a `Procfile`, and `render.yaml` are ready.

---

## Step 1 — Build the data bundle (once, locally)

In your project folder, with the model already trained:

```bash
python build_artifacts.py
```

This creates `website/data/` (scores, returns, backtest). Confirm the folder
exists — it must be committed to your repo (it's what the live site serves).

## Step 2 — Put the project on GitHub

Create a free GitHub account if you don't have one, then in your project folder:

```bash
git init
git add .
git commit -m "Smart Investment Portfolio System"
```

The included `.gitignore` keeps the huge CSV, the model file, and local databases
out of the repo — the live site doesn't need them, only the small bundle.

Create a new empty repository on github.com, then push (copy the two lines GitHub
shows you, they look like):

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git branch -M main
git push -u origin main
```

## Step 3 — Deploy on Render (free)

1. Sign up at **render.com** (free) and connect your GitHub.
2. Click **New +** → **Blueprint**.
3. Pick your repository. Render finds `render.yaml` and shows a web service plus a
   free PostgreSQL database.
4. Click **Apply**. Render builds and deploys (first build takes a few minutes).
5. When it's live, you get a public URL like
   `https://smart-investment-portfolio.onrender.com` — that's your link.

That's it. The blueprint wires the database, generates the secret key, and sets
the start command automatically.

---

## Good to know

- **First load after idle is slow.** On the free plan the service sleeps after
  ~15 minutes of no traffic; the next visit takes ~30–60 seconds to wake up. Open
  the link a minute before you show it so it's already awake.
- **Don't want a database yet?** Delete the `databases:` block and the
  `DATABASE_URL` line from `render.yaml`. The site then uses SQLite and works
  immediately — the only difference is saved portfolios reset when the free
  instance restarts. The core (generate / forecast / compare within a session)
  works either way.
- **Free Postgres** on Render is time-limited (it expires after a period). For a
  project demo that's fine; for anything longer, upgrade the database or point
  `DATABASE_URL` at another free Postgres (e.g. Neon or Supabase).
- **Updating the site later:** `git push` again and Render redeploys automatically.
- **Refreshing the data** (e.g. new prices): re-run `train_model.py` and
  `build_artifacts.py` locally, commit, and push.

## Manual alternative (no blueprint)

If you'd rather set it up by hand: New + → **Web Service** → connect the repo, then
set **Build command** to `pip install -r website/requirements.txt` and **Start
command** to
`gunicorn --chdir website --preload -w 2 -b 0.0.0.0:$PORT server:app`.
Add an environment variable `SECRET_KEY` (any long random string). Optionally add a
PostgreSQL instance and set `DATABASE_URL` to its connection string.
