"""
server.py  -  Flask web server for the Smart Investment Portfolio System
------------------------------------------------------------------------
The deployable website. Serves the frontend and exposes the model through a
JSON API. Each visitor gets an anonymous session (a signed cookie), so saved
portfolios are private to them without any login or personal data.

Local run:   cd website && python server.py   -> http://localhost:5000
Production:  gunicorn --preload -w 2 server:app   (see DEPLOY.md)
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, session
import web_service

app = Flask(__name__)
# secret key signs the session cookie. Set a real one via env in production.
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-me")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

RISKS = {"Low", "Medium", "High"}


def uid():
    """Stable anonymous id for this visitor, stored in the signed cookie."""
    if "uid" not in session:
        session["uid"] = uuid.uuid4().hex
    return session["uid"]


@app.route("/")
def home():
    web_service.market_snapshot()  # starts a background refresh if data is stale (non-blocking)
    return render_template("index.html", as_of=web_service.as_of())


@app.route("/api/portfolio", methods=["POST"])
def api_portfolio():
    data = request.get_json(force=True) or {}
    try:
        amount = int(float(data.get("amount", 10000)))
        risk = str(data.get("risk", "Medium")).title()
        horizon = int(data.get("horizon", 60))
    except (TypeError, ValueError):
        return jsonify(error="Please enter a valid amount, risk, and horizon."), 400
    if amount < 100 or amount > 10_000_000:
        return jsonify(error="Amount must be between $100 and $10,000,000."), 400
    if risk not in RISKS:
        return jsonify(error="Risk must be Low, Medium, or High."), 400
    horizon = max(5, min(252, horizon))
    return jsonify(web_service.build_portfolio(amount, risk, horizon))


@app.route("/api/market")
def api_market():
    resp = jsonify(web_service.market_snapshot())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/profile")
def api_profile():
    risk = str(request.args.get("risk", "Medium")).title()
    if risk not in RISKS:
        return jsonify(error="Risk must be Low, Medium, or High."), 400
    resp = jsonify(web_service.build_profile(risk))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/backtest")
def api_backtest():
    return jsonify(web_service.get_backtest())


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True) or {}
    if "holdings" not in data:
        return jsonify(error="Generate a portfolio before saving."), 400
    pid = web_service.save_portfolio(uid(), data)
    return jsonify(id=pid, saved=web_service.list_saved(uid()))


@app.route("/api/saved")
def api_saved():
    return jsonify(web_service.list_saved(uid()))


@app.route("/api/clear", methods=["POST"])
def api_clear():
    web_service.clear_saved(uid())
    return jsonify(saved=[])


@app.route("/healthz")
def healthz():
    return "ok"


# load the model/bundle once, when the app object is created (works under gunicorn too)
print("Loading data bundle ...")
web_service.boot()
print("Ready.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
