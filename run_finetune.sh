#!/usr/bin/env bash
set -euo pipefail

# ─── Config ───────────────────────────────────────────────────────────────────
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
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

SPEAKER_NAME="turkish_speaker"
BATCH_SIZE=2
LR=2e-5
EPOCHS=3
TARGET_LUFS=-18.0
# ──────────────────────────────────────────────────────────────────────────────

echo "============================================"
echo " Qwen3-TTS Fine-tuning Pipeline"
echo "============================================"

# 1. HF token
if [ -f ".env" ] && grep -q "HF_TOKEN" .env; then
    export HF_TOKEN=$(grep "HF_TOKEN" .env | cut -d '=' -f2 | tr -d '"')
    echo "[1/6] Loaded HF_TOKEN from .env"
else
    read -rsp "Enter your Hugging Face token: " HF_TOKEN
    echo
    export HF_TOKEN
    echo "[1/6] HF_TOKEN set"
fi

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# 2. Install dependencies
echo ""
echo "[2/6] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -e "$(dirname "$0")"
pip install -q pyloudnorm datasets soundfile
# flash-attn for faster training (optional but recommended on A100/H100)
pip install -q flash-attn --no-build-isolation || echo "  flash-attn install failed, continuing without it (slower training)"

# 3. Download dataset and convert to JSONL
echo ""
echo "[3/6] Downloading dataset and extracting audio..."
mkdir -p "$AUDIO_RAW_DIR" "$AUDIO_NORM_DIR" "$DATA_DIR"

python3 - <<PYEOF
import os, json, soundfile as sf
from datasets import load_dataset

hf_token = os.environ["HF_TOKEN"]
audio_dir = "$AUDIO_RAW_DIR"
out_jsonl  = "$RAW_JSONL"

print("  Loading dataset from HuggingFace (this may take a while)...")
ds = load_dataset("$HF_DATASET", split="train", token=hf_token)

ref_audio_path = None
lines = []

for i, sample in enumerate(ds):
    audio_data  = sample["audio"]      # dict: {"array": np.ndarray, "sampling_rate": int}
    text        = sample["text"]
    speaker_id  = sample.get("speaker_id", "speaker")

    wav_path = os.path.join(audio_dir, f"{i:06d}.wav")
    sf.write(wav_path, audio_data["array"], audio_data["sampling_rate"])

    # Use the very first sample as the single ref_audio for all (recommended)
    if ref_audio_path is None:
        ref_audio_path = wav_path

    lines.append({"audio": wav_path, "text": text, "ref_audio": ref_audio_path, "speaker_id": speaker_id})

    if (i + 1) % 1000 == 0:
        print(f"  Extracted {i + 1}/{len(ds)} samples...")

with open(out_jsonl, "w") as f:
    for line in lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"  Done. {len(lines)} samples written to {out_jsonl}")
PYEOF

# 4. Normalize loudness
echo ""
echo "[4/6] Normalizing loudness to ${TARGET_LUFS} LUFS..."
python3 finetuning/normalize.py \
    --input_jsonl  "$RAW_JSONL" \
    --output_jsonl "$NORM_JSONL" \
    --output_audio_dir "$AUDIO_NORM_DIR" \
    --target_lufs "$TARGET_LUFS"

# 5. Prepare data (extract audio codes)
echo ""
echo "[5/6] Extracting audio codes (prepare_data)..."
python3 finetuning/prepare_data.py \
    --device "$DEVICE" \
    --tokenizer_model_path "$TOKENIZER_MODEL_PATH" \
    --input_jsonl  "$NORM_JSONL" \
    --output_jsonl "$TRAIN_JSONL"

# 6. Fine-tune
echo ""
echo "[6/6] Starting fine-tuning..."
cd finetuning
accelerate launch sft_12hz.py \
    --init_model_path  "$INIT_MODEL_PATH" \
    --output_model_path "../$OUTPUT_DIR" \
    --train_jsonl      "../$TRAIN_JSONL" \
    --batch_size       "$BATCH_SIZE" \
    --lr               "$LR" \
    --num_epochs       "$EPOCHS" \
    --speaker_name     "$SPEAKER_NAME"

echo ""
echo "============================================"
echo " Done! Checkpoints saved to: $OUTPUT_DIR/"
echo "============================================"
