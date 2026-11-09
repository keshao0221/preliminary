set -euo pipefail
export MINIMAX_REPRO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec python3 -u "$MINIMAX_REPRO_ROOT/tools/reproduce.py" "$@"
