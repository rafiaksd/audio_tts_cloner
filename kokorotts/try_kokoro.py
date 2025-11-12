from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import os
import numpy as np

output_audio_name = "advancement of TTS recently"

audio_folder = "audios/"
os.makedirs(audio_folder, exist_ok=True)

pipeline = KPipeline(lang_code='a')

text = '''
Text-to-Speech technology has come a long way in recent years! Modern TTS systems can now produce voices that sound natural, expressive, and even emotional.
Once you start experimenting with them, you’ll see how every model has its own unique tone and rhythm — and that makes it easier not only to create lifelike speech,
but also to explore the exciting connection between language, sound, and human expression.
'''

all_audios = []

generator = pipeline(text, voice='am_puck')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    segment_path = os.path.join(audio_folder, f"{i}.wav")
    sf.write(segment_path, audio, 24000)
    all_audios.append(audio)

# === Combine all audio clips ===
if all_audios:
    combined = np.concatenate(all_audios)
    combined_path = os.path.join(audio_folder, f"{output_audio_name}.wav")
    sf.write(combined_path, combined, 24000)
    print(f"✅ Combined audio saved to: {combined_path}")
    display(Audio(data=combined, rate=24000, autoplay=True))

    # === Delete individual temporary segment files ===
    for file in os.listdir(audio_folder):
        if file.endswith(".wav") and file != f"{output_audio_name}.wav":
            os.remove(os.path.join(audio_folder, file))
    print("🧹 Deleted individual segment files.")
else:
    print("⚠️ No audio segments were generated.")
