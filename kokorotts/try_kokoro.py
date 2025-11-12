from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import os
import numpy as np

output_audio_name = "interesting shark fact"
audio_folder = "audios/"
os.makedirs(audio_folder, exist_ok=True)

pipeline = KPipeline(lang_code='a')

text = '''
Sharks have been swimming in the oceans for over 400 million years! That is even before trees grew on land. Some sharks can smell blood from miles away, and the biggest shark, the whale shark, can grow longer than a bus. Sharks are amazing hunters, but they help keep the oceans healthy by eating sick fish.
'''

# === Track temporary segment paths ===
temp_files = []
all_audios = []

# am_echo, am_puck, 
generator = pipeline(text, voice='am_puck')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    
    segment_path = os.path.join(audio_folder, f"temp_{i}.wav")
    sf.write(segment_path, audio, 24000)
    temp_files.append(segment_path)
    all_audios.append(audio)

# === Combine all audio clips ===
if all_audios:
    combined = np.concatenate(all_audios)
    combined_path = os.path.join(audio_folder, f"{output_audio_name}.wav")
    sf.write(combined_path, combined, 24000)
    print(f"✅ Combined audio saved to: {combined_path}")
    display(Audio(data=combined, rate=24000, autoplay=True))

    # === Delete only the temporary files created in this run ===
    for file_path in temp_files:
        os.remove(file_path)
    print("🧹 Deleted temporary segment files.")
else:
    print("⚠️ No audio segments were generated.")
