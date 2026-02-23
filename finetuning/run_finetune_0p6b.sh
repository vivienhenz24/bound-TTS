#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------
# Config (override via env vars)
# ------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

DEVICE="${DEVICE:-cuda:0}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-Qwen/Qwen3-TTS-Tokenizer-12Hz}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"

RAW_JSONL="${RAW_JSONL:-${SCRIPT_DIR}/train_raw.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-${SCRIPT_DIR}/train_with_codes.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_0p6b}"

BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker_1}"

# If 1: always regenerate TRAIN_JSONL from RAW_JSONL.
# If 0: only generate TRAIN_JSONL when it does not exist.
FORCE_PREPARE="${FORCE_PREPARE:-0}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ "${USE_VENV}" == "1" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

log "Installing/refreshing dependencies"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

# Install torch only if missing in the active environment.
if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
  log "PyTorch not found, installing torch + torchaudio"
  "${PYTHON_BIN}" -m pip install torch torchaudio
fi

"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
"${PYTHON_BIN}" -m pip install safetensors tensorboard

mkdir -p "${OUTPUT_DIR}"

pushd "${SCRIPT_DIR}" >/dev/null

if [[ "${FORCE_PREPARE}" == "1" || ! -f "${TRAIN_JSONL}" ]]; then
  if [[ ! -f "${RAW_JSONL}" ]]; then
    echo "Missing RAW_JSONL: ${RAW_JSONL}" >&2
    echo "Provide a raw jsonl or set TRAIN_JSONL to an existing prepared file." >&2
    exit 1
  fi

  log "Preparing data: ${RAW_JSONL} -> ${TRAIN_JSONL}"
  "${PYTHON_BIN}" prepare_data.py \
    --device "${DEVICE}" \
    --tokenizer_model_path "${TOKENIZER_MODEL_PATH}" \
    --input_jsonl "${RAW_JSONL}" \
    --output_jsonl "${TRAIN_JSONL}"
else
  log "Using existing prepared file: ${TRAIN_JSONL}"
fi

log "Starting fine-tuning"
"${PYTHON_BIN}" sft_12hz.py \
  --init_model_path "${INIT_MODEL_PATH}" \
  --output_model_path "${OUTPUT_DIR}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_epochs "${EPOCHS}" \
  --speaker_name "${SPEAKER_NAME}"

popd >/dev/null

log "Finished. Checkpoints are under: ${OUTPUT_DIR}"
