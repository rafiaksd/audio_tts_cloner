# install espeak-ng
# pip install torch torchaudio
# pip install phonemizer sudachipy sudachidict_full kanjize

import winsound

import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
print(f"============ 1 Started model loading")
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

print(f"============ 2 Started loading audio to torchaudio")
wav, sampling_rate = torchaudio.load("js_mastery_voice.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

text_to_voice = """
Text-to-Speech technology has come a long way in recent years! Modern TTS systems can now produce voices that sound natural, expressive, and even emotional.
Once you start experimenting with them, you’ll see how every model has its own unique tone and rhythm — and that makes it easier not only to create lifelike speech,
but also to explore the exciting connection between language, sound, and human expression.
"""

print(f"============ 3 Making cond dict...")
cond_dict = make_cond_dict(text=text_to_voice, speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

print(f"============ 4 Generating audio")
wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("arabic_intro.wav", wavs[0], model.autoencoder.sampling_rate)

winsound.Beep(1000,500)