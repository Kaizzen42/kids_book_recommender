#!/usr/bin/env bash
set -euo pipefail

# -------- load env --------
if [[ -f ".env" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
else
  echo "ERROR: .env not found. Create one with BOOKS_API_KEY/NG_USER/NG_PASS/SERVER_HOST/SERVER_PORT." >&2
  exit 1
fi

# sanity checks (don’t print secrets)
for var in BOOKS_API_KEY NG_USER NG_PASS SERVER_HOST SERVER_PORT; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: $var is unset in .env" >&2
    exit 1
  fi
done

echo "[serve] HOST=$SERVER_HOST  PORT=$SERVER_PORT"
echo "[serve] ngrok Basic Auth user: $NG_USER"
echo "[serve] opening ngrok in a NEW Terminal window…"

# -------- activate venv --------
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD"

# -------- launch ngrok in a new Terminal window (macOS) --------
# We pass all commands explicitly because a new Terminal session won't inherit this shell's env.
osascript <<OSA
tell application "iTerm"
  create window with default profile
  tell current window's current session
    write text "cd '$PWD'; \
      test -f .venv/bin/activate && source .venv/bin/activate || true; \
      export NG_USER='$NG_USER'; \
      export NG_PASS='$NG_PASS'; \
      export SERVER_PORT='$SERVER_PORT'; \
      ngrok http \$SERVER_PORT --basic-auth \"\${NG_USER}:\${NG_PASS}\""
  end tell
  activate
end tell
OSA

echo "[serve] ngrok window opened. Tip: local web UI at http://127.0.0.1:4040 shows the public URL."

# -------- start FastAPI (foreground) --------
python -m uvicorn src.server:app --host "$SERVER_HOST" --port "$SERVER_PORT" --reload
echo "BOOKS_API_KEY:$len(BOOKS_API_KEY)"
