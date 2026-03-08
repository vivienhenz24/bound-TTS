# coding=utf-8
import argparse
import json
import os

import librosa
import numpy as np
import soundfile as sf

TARGET_PEAK_DBFS = -1.0


def normalize_peak(audio: np.ndarray, target_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-9:  # silence, skip
        return audio
    target_amplitude = 10 ** (target_dbfs / 20.0)
    return audio * (target_amplitude / peak)


def main():
    parser = argparse.ArgumentParser(
        description="Peak-normalize audio files referenced in a JSONL dataset."
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="JSONL with updated 'audio' paths pointing to normalized files.")
    parser.add_argument("--output_audio_dir", type=str, required=True,
                        help="Directory to write normalized wav files.")
    parser.add_argument("--target_dbfs", type=float, default=TARGET_PEAK_DBFS,
                        help="Target peak level in dBFS (default: -1.0)")
    args = parser.parse_args()

    os.makedirs(args.output_audio_dir, exist_ok=True)

    lines = [json.loads(l.strip()) for l in open(args.input_jsonl)]

    with open(args.output_jsonl, "w") as out_f:
        for i, line in enumerate(lines):
            src_path = line["audio"]
            audio, sr = librosa.load(src_path, sr=None, mono=True)
            audio = normalize_peak(audio, args.target_dbfs)

            basename = os.path.splitext(os.path.basename(src_path))[0]
            dst_path = os.path.join(args.output_audio_dir, f"{basename}_norm.wav")
            sf.write(dst_path, audio, sr)

            line["audio"] = dst_path
            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(lines)}")

    print(f"Done. Normalized {len(lines)} files -> {args.output_audio_dir}")


if __name__ == "__main__":
    main()
