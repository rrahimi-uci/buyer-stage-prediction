#!/usr/bin/env bash
#
# stop.sh — stop the stack.
#
#   ./stop.sh             stop & remove the containers   (KEEPS all data)
#   ./stop.sh --reset     also remove the named volumes  (WIPES Postgres + MLflow data)
#
# Note: --reset wipes the docker volumes (pg_platform, pg_serving, mlruns). Offline feature
# matrices under ./data and Dagster's ./.dagster stay on the host — delete those by hand for a
# truly clean slate.
set -euo pipefail
cd "$(dirname "$0")"

if [ -t 1 ]; then
  BOLD=$'\e[1m'; DIM=$'\e[2m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
  BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

PROFILES=(--profile core --profile dashboard)
RESET_VOLUMES=0
for arg in "$@"; do
  case "$arg" in
    --reset|-v|--volumes) RESET_VOLUMES=1 ;;
    -h|--help)  sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "${RED}unknown option: $arg${RESET} (try --help)" >&2; exit 2 ;;
  esac
done

if ! docker info >/dev/null 2>&1; then
  echo "${YELLOW}Docker isn't running — nothing to stop.${RESET}"
  exit 0
fi

if [ "$RESET_VOLUMES" = "1" ]; then
  echo "${YELLOW}${BOLD}▶ Stopping and WIPING volumes (Postgres + MLflow data will be lost)…${RESET}"
  docker compose "${PROFILES[@]}" down -v --remove-orphans
  echo "${DIM}(host ./data and ./.dagster were left in place — remove them manually if you want a full reset)${RESET}"
else
  echo "${BOLD}▶ Stopping stack (data preserved)…${RESET}"
  docker compose "${PROFILES[@]}" down --remove-orphans
fi

echo "${GREEN}✓ Stopped.${RESET}"
