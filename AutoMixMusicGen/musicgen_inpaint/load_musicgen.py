from audiocraft.models import MusicGen

def load_small_model():
    model = MusicGen.get_pretrained("facebook/musicgen-small")

    for p in model.lm.parameters(): p.data = p.data.float()
    for p in model.compression_model.parameters(): p.data = p.data.float()

    return model
