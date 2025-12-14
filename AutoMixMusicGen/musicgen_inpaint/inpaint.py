import torch
import torchaudio
from IPython.display import Audio, display

def inpaint_audio(model, left_path, right_path, left_sec=4, right_sec=4, mask_sec=4, device="cuda"):
    L, _ = torchaudio.load(left_path)
    R, _ = torchaudio.load(right_path)

    L_wave = L.to("cuda")
    R_wave = R.to("cuda")

    display(Audio(L.numpy(), rate=model.sample_rate))
    display(Audio(R.numpy(), rate=model.sample_rate))

    codec = model.compression_model.to(device)

    # encode
    with torch.no_grad():
        L_code, _ = codec.encode(L_wave.unsqueeze(0))
        R_code, _ = codec.encode(R_wave.unsqueeze(0))

    L_code = L_code[0].unsqueeze(0).long()  # (1,K,T)
    R_code = R_code[0].unsqueeze(0).long()

    # convert sec → tokens
    L_n = int(left_sec / 0.02)
    R_n = int(right_sec / 0.02)
    M_n = int(mask_sec  / 0.02)

    L_tok = L_code[:, :, -L_n:]
    R_tok = R_code[:, :, :R_n]

    mask_id = int(model.lm.special_token_id or 0)
    mask = torch.full((1, L_code.shape[1], M_n),
                      mask_id, dtype=torch.long).to(device)

    seq = torch.cat([L_tok, mask, R_tok], dim=2).to(device)

    # LM prediction
    attrs, _ = model._prepare_tokens_and_attributes([None], None)
    with torch.no_grad():
        out = model.lm.compute_predictions(seq, conditions=attrs)

    pred = out.logits.argmax(-1)

    # Inpaint mask only
    seq[:, :, L_n:L_n+M_n] = pred[:, :, L_n:L_n+M_n]

    # decode to audio
    with torch.no_grad():
        wav = codec.decode(seq)[0]

    return wav.cpu().numpy()
