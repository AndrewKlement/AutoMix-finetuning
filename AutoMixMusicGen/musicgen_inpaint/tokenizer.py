import os
import torch
import torchaudio

def tokenize_dataset(model, in_dir="dataset/chunks", out_dir="dataset/tokenized"):
    codec = model.compression_model.to("cuda")
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.endswith(".wav"): continue

        path = os.path.join(in_dir, fname)
        wav, _ = torchaudio.load(path)
        wav = wav.mean(0).unsqueeze(0).unsqueeze(0).to("cuda")  # mono -> (1,1,T)

        with torch.no_grad():
            enc, _ = codec.encode(wav)

        codes = enc[0].cpu()  # (K,T)
        torch.save(codes, os.path.join(out_dir, fname.replace(".wav", ".pt")))

    print("[Tokenizer] Finished encoding all audio.")
