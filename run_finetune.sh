#!/usr/bin/env bash
set -euo pipefail

# ─── Config ───────────────────────────────────────────────────────────────────
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
HF_DATASET="vsqrd/turkish-tts-dataset-temp"
DEVICE="cuda:0"

DATA_DIR="data"
AUDIO_RAW_DIR="$DATA_DIR/audio_raw"
AUDIO_NORM_DIR="$DATA_DIR/audio_normalized"
RAW_JSONL="$DATA_DIR/train_raw.jsonl"
NORM_JSONL="$DATA_DIR/train_normalized.jsonl"
TRAIN_JSONL="$DATA_DIR/train_with_codes.jsonl"
OUTPUT_DIR="output"

SPEAKER_NAME="female_speaker"
FILTER_SPEAKER_ID="female_speaker"
TEST_TEXT="Merhaba, bu bir ses testi cümlesidir."
BATCH_SIZE=2
LR=2e-5
EPOCHS=10
TARGET_DBFS=-1.0
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_step() { echo ""; echo "[$(date '+%Y-%m-%d %H:%M:%S')] ══════════ $* ══════════"; }

log "============================================"
log " Qwen3-TTS Fine-tuning Pipeline"
log " Log file: $LOG_FILE"
log "============================================"

# 1. HF token
log_step "1/6 — HF Authentication"
if [ -f ".env" ] && grep -q "HF_TOKEN" .env; then
    export HF_TOKEN=$(grep "HF_TOKEN" .env | cut -d '=' -f2 | tr -d '"')
    log "Loaded HF_TOKEN from .env"
else
    read -rsp "Enter your Hugging Face token: " HF_TOKEN
    echo
    export HF_TOKEN
    log "HF_TOKEN set from prompt"
fi

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
log "HuggingFace login OK"

# 2. Install dependencies
log_step "2/6 — Installing dependencies"
pip install -q --upgrade pip
pip install -q -e "$(dirname "$0")"
pip install -q pyloudnorm datasets soundfile
apt-get install -y -q sox 2>/dev/null || log "WARNING: sox install failed"
log "Core dependencies installed"
pip install -q hf_transfer
TMPDIR=/tmp PIP_CACHE_DIR=/tmp/pip-cache pip install -q flash-attn --no-build-isolation \
    && log "flash-attn installed" \
    || log "WARNING: flash-attn install failed — continuing without it (slower training)"

# 3. Download dataset and convert to JSONL
log_step "3/6 — Downloading dataset and extracting audio"
mkdir -p "$AUDIO_RAW_DIR" "$AUDIO_NORM_DIR" "$DATA_DIR"

if [ -f "$RAW_JSONL" ]; then
    log "SKIP: $RAW_JSONL already exists"
else
log "Saving audio to: $AUDIO_RAW_DIR"
log "Output JSONL: $RAW_JSONL"

python3 - <<PYEOF
import io, os, json, time
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio

hf_token  = os.environ["HF_TOKEN"]
audio_dir = "$AUDIO_RAW_DIR"
out_jsonl = "$RAW_JSONL"

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from HuggingFace...")
# decode=False to get raw bytes — avoids torchcodec dependency
ds = load_dataset("$HF_DATASET", split="train", token=hf_token).cast_column("audio", Audio(decode=False))
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset loaded: {len(ds)} samples")

ref_audio_path = None
lines = []
t0 = time.time()

for i, sample in enumerate(ds):
    audio_data = sample["audio"]   # {"bytes": ..., "path": ...}
    text       = sample["text"]
    speaker_id = sample.get("speaker_id", "speaker")

    # Decode bytes with soundfile
    raw = audio_data.get("bytes") or open(audio_data["path"], "rb").read()
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)

    wav_path = os.path.join(audio_dir, f"{i:06d}.wav")
    sf.write(wav_path, wav, sr)

    if ref_audio_path is None:
        ref_audio_path = wav_path
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using ref_audio: {ref_audio_path}")

    lines.append({"audio": wav_path, "text": text, "ref_audio": ref_audio_path, "speaker_id": speaker_id})

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(ds) - i - 1) / rate
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Extracted {i+1}/{len(ds)} samples "
              f"({rate:.1f} samples/s, ETA {eta/60:.1f} min)")

with open(out_jsonl, "w") as f:
    for line in lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

elapsed = time.time() - t0
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done. {len(lines)} samples written to {out_jsonl} in {elapsed/60:.1f} min")
PYEOF

log "Dataset extraction complete"
fi  # end skip check

# 3b. Filter to target speaker
FILTERED_JSONL="$DATA_DIR/train_raw_filtered.jsonl"
if [ -f "$FILTERED_JSONL" ]; then
    log "SKIP: $FILTERED_JSONL already exists"
else
    log "Filtering to speaker: $FILTER_SPEAKER_ID"
    python3 -c "
import json
lines = [l for l in open('$RAW_JSONL') if json.loads(l)['speaker_id'] == '$FILTER_SPEAKER_ID']
# Set ref_audio to the first female sample for all entries
first = json.loads(lines[0])['audio']
out = []
for l in lines:
    d = json.loads(l)
    d['ref_audio'] = first
    out.append(json.dumps(d, ensure_ascii=False))
open('$FILTERED_JSONL', 'w').write('\n'.join(out) + '\n')
print(f'Kept {len(out)} samples for $FILTER_SPEAKER_ID')
"
    log "Filtering complete"
fi
RAW_JSONL="$FILTERED_JSONL"

# 4. Normalize loudness
log_step "4/6 — Peak-normalizing to ${TARGET_DBFS} dBFS"
if [ -f "$NORM_JSONL" ]; then
    log "SKIP: $NORM_JSONL already exists"
else
    log "Input:  $RAW_JSONL"
    log "Output: $NORM_JSONL"
    python3 finetuning/normalize.py \
        --input_jsonl      "$RAW_JSONL" \
        --output_jsonl     "$NORM_JSONL" \
        --output_audio_dir "$AUDIO_NORM_DIR" \
        --target_dbfs      "$TARGET_DBFS"
    log "Loudness normalization complete"
fi

# 5. Prepare data (extract audio codes)
log_step "5/6 — Extracting audio codes"
if [ -f "$TRAIN_JSONL" ]; then
    log "SKIP: $TRAIN_JSONL already exists"
else
    log "Input:  $NORM_JSONL"
    log "Output: $TRAIN_JSONL"
    python3 finetuning/prepare_data.py \
        --device               "$DEVICE" \
        --tokenizer_model_path "$TOKENIZER_MODEL_PATH" \
        --input_jsonl          "$NORM_JSONL" \
        --output_jsonl         "$TRAIN_JSONL"
    log "Audio code extraction complete"
fi

# 6. Fine-tune
log_step "6/6 — Fine-tuning"
log "Model:        $INIT_MODEL_PATH"
log "Output dir:   $OUTPUT_DIR"
log "Batch size:   $BATCH_SIZE"
log "LR:           $LR"
log "Epochs:       $EPOCHS"
log "Speaker name: $SPEAKER_NAME"

PYTHONPATH="finetuning:${PYTHONPATH:-}" accelerate launch finetuning/sft_12hz.py \
    --init_model_path  "$INIT_MODEL_PATH" \
    --output_model_path "$OUTPUT_DIR" \
    --train_jsonl       "$TRAIN_JSONL" \
    --batch_size        "$BATCH_SIZE" \
    --lr                "$LR" \
    --num_epochs        "$EPOCHS" \
    --speaker_name      "$SPEAKER_NAME" \
    --test_text         "$TEST_TEXT"

log ""
log "============================================"
log " Done! Checkpoints saved to: $OUTPUT_DIR/"
log " Full log: $LOG_FILE"
log "============================================"
