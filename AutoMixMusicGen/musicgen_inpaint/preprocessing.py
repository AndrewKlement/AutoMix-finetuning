# audio/preprocessing.py
import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 32000
CHUNK_DURATION = 10.0

def collect_audio_files(root):
    root = Path(root)
    return [p for folder in root.iterdir() for p in folder.iterdir()]

def split_into_chunks(path, sample_rate=SAMPLE_RATE, duration=CHUNK_DURATION):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    chunk_len = int(sample_rate * duration)
    chunks = []

    for start in range(0, len(audio), chunk_len):
        block = audio[start:start+chunk_len]

        if len(block) < chunk_len * 0.5:
            continue

        if len(block) < chunk_len:
            block = np.pad(block, (0, chunk_len - len(block)))

        chunks.append(block)

    return chunks

def preprocess_dataset(raw_dir="dataset/raw", out_dir="dataset/chunks"):
    os.makedirs(out_dir, exist_ok=True)
    audio_files = collect_audio_files(raw_dir)

    for path in audio_files:
        chunks = split_into_chunks(path)

        for i, c in enumerate(chunks):
            out_name = f"{path.stem}_chunk{i:04d}.wav"
            sf.write(Path(out_dir) / out_name, c, SAMPLE_RATE)

    print("[Preprocessing] Complete.")
