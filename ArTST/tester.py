from transformers import pipeline
from datasets import load_dataset
import torch
import soundfile as sf

# Load TTS pipeline
print(f"\n\n =============== Loading model...")
synthesiser = pipeline("text-to-speech", "MBZUAI/speecht5_tts_clartts_ar")

print(f"\n\n =============== Loading embeding...")
# Load a speaker embedding from dataset (pretrained voice style)
embeddings_dataset = load_dataset("herwoww/arabic_xvector_embeddings", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[105]["speaker_embeddings"]).unsqueeze(0)

# Your Arabic text
text_arabic = """
تعلم اللغة العربية ليس صعبًا كما يظن الكثير من الناس! العربية لغة منظمة ولها قواعد واضحة. عندما تبدأ بالتعلم، ستلاحظ أن كل شيء يتبع نظامًا محددًا، وهذا يجعل من السهل تعلم اللغة وفهم معاني الكلمات والجمل الجميلة.
"""

print(f"\n\n =============== Generating speech...")
speech = synthesiser(text_arabic, forward_params={"speaker_embeddings": speaker_embedding})

# Save as WAV
sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
print("✅ Speech saved to speech.wav")
