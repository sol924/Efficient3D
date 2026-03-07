#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMA_FILE="${ROOT_DIR}/transformers/src/transformers/models/llama/modeling_llama.py"
BACKUP_FILE="${LLAMA_FILE}.bak_commented_adaptive"

if [[ ! -f "${LLAMA_FILE}" ]]; then
  echo "[ERROR] Cannot find: ${LLAMA_FILE}" >&2
  exit 1
fi

restore_file() {
  if [[ -f "${BACKUP_FILE}" ]]; then
    mv -f "${BACKUP_FILE}" "${LLAMA_FILE}"
    echo "[INFO] Restored original modeling_llama.py"
  fi
}
trap restore_file EXIT

cp -f "${LLAMA_FILE}" "${BACKUP_FILE}"

python - "${LLAMA_FILE}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines(True)

start = None
end = None

for i, line in enumerate(lines):
    if line.startswith("# @add_start_docstrings("):
        start = i
        break

if start is None:
    raise SystemExit("Cannot find commented adaptive LlamaModel start marker.")

for i in range(start + 1, len(lines)):
    if lines[i].startswith("class LlamaForCausalLM"):
        end = i
        break

if end is None:
    raise SystemExit("Cannot find adaptive block end marker (class LlamaForCausalLM).")

for i in range(start, end):
    line = lines[i]
    if line.startswith("# "):
        lines[i] = line[2:]
    elif line.startswith("#"):
        lines[i] = line[1:]

path.write_text("".join(lines), encoding="utf-8")
print(f"[INFO] Enabled commented adaptive block: lines {start+1}..{end}")
PY

python -m py_compile "${LLAMA_FILE}"
echo "[INFO] Patched file compiles successfully."

if [[ "$#" -eq 0 ]]; then
  echo "[INFO] No command provided, running default:"
  echo "       bash ${SCRIPT_DIR}/batch_eval_efficient3d_pred_attn.sh"
  bash "${SCRIPT_DIR}/batch_eval_efficient3d_pred_attn.sh"
else
  echo "[INFO] Running custom command: $*"
  "$@"
fi
