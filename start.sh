#!/usr/bin/env bash
#
# start.sh — bring up the full stack (core services + Streamlit dashboard) and wait until healthy.
#
#   ./start.sh            start everything (builds images on first run, reuses after)
#   ./start.sh --demo     also seed example data + run the pipeline once (fresh, populated demo)
#   ./start.sh --build    force a rebuild of the images (after Dockerfile/dependency changes)
#
# Ports may be remapped locally via docker-compose.override.yml (e.g. to dodge a clashing stack),
# so the URLs printed at the end are read from the containers' ACTUAL published ports.
set -euo pipefail
cd "$(dirname "$0")"

if [ -t 1 ]; then
  BOLD=$'\e[1m'; DIM=$'\e[2m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
  BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

PROFILES=(--profile core --profile dashboard)
DEMO=0
BUILD=0
for arg in "$@"; do
  case "$arg" in
    --demo)      DEMO=1 ;;
    --build)     BUILD=1 ;;
    -h|--help)   sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "${RED}unknown option: $arg${RESET} (try --help)" >&2; exit 2 ;;
  esac
done

if ! docker info >/dev/null 2>&1; then
  echo "${RED}✗ Docker isn't running.${RESET} Start Docker Desktop and retry." >&2
  exit 1
fi

# --- bring the stack up -------------------------------------------------------
echo "${BOLD}▶ Starting stack (core + dashboard)…${RESET}"
if [ "$BUILD" = "1" ]; then
  docker compose "${PROFILES[@]}" up -d --build
else
  docker compose "${PROFILES[@]}" up -d   # auto-builds missing images; bind mounts pick up code edits
fi

# --- wait for health ----------------------------------------------------------
url_for() {  # url_for <service> <container-port> -> http://localhost:<host-port>  (or "")
  local p; p=$(docker compose port "$1" "$2" 2>/dev/null | awk -F: 'NR==1{print $NF}')
  [ -n "$p" ] && echo "http://localhost:$p" || echo ""
}

wait_http() {  # wait_http <label> <url> <health-path> [tries]
  local label=$1 base=$2 path=$3 tries=${4:-45}
  printf "  %-11s" "$label"
  if [ -z "$base" ]; then echo "${YELLOW}not started${RESET}"; return 1; fi
  for _ in $(seq 1 "$tries"); do
    local code; code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$base$path" 2>/dev/null || true)
    if [ "$code" = "200" ]; then echo "${GREEN}ready${RESET}  ${DIM}$base${RESET}"; return 0; fi
    sleep 2
  done
  echo "${YELLOW}still starting${RESET}  ${DIM}$base${RESET}"; return 1
}

echo "${BOLD}▶ Waiting for services…${RESET}"
DASH_URL=$(url_for dashboard 8501)
API_URL=$(url_for api 8000)
MLF_URL=$(url_for mlflow 5000)
DAG_URL=$(url_for dagster 3000)
wait_http "dashboard" "$DASH_URL" "/_stcore/health" || true
wait_http "api"       "$API_URL"  "/healthz"        || true
wait_http "mlflow"    "$MLF_URL"  "/health"         || true
wait_http "dagster"   "$DAG_URL"  "/server_info"    || true

# --- optional: seed + run for a populated demo --------------------------------
if [ "$DEMO" = "1" ]; then
  echo "${BOLD}▶ Seeding example data…${RESET}"
  docker compose run --rm dagster python -m examples.buyer_stage.seed_raw
  echo "${BOLD}▶ Running the pipeline once (FLAML train → score → load)…${RESET}"
  docker compose run --rm dagster automl-run
fi

# --- summary ------------------------------------------------------------------
echo
echo "${BOLD}${GREEN}✓ Stack is up.${RESET}"
printf "  %-11s %s\n" "Dashboard" "${DASH_URL:-(not running)}"
printf "  %-11s %s\n" "Dagster"   "${DAG_URL:-(not running)}"
printf "  %-11s %s\n" "MLflow"    "${MLF_URL:-(not running)}"
printf "  %-11s %s\n" "API docs"  "${API_URL:+$API_URL/docs}"
echo
echo "${DIM}Tip: ./start.sh --demo seeds + runs once so the dashboard opens populated."
echo "     Stop with ./stop.sh  (add --reset to also wipe Postgres + MLflow data).${RESET}"
