from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile

text_to_convert = """
تعلّمَ اللُّغةَ العَرَبِيّةَ لَيسَ صَعبًا كما يَظُنُّ الكَثيرُ مِنَ النّاسِ! العَرَبِيّةُ لُغةٌ مُنَظَّمَةٌ وَلَها قَواعِدُ واضِحَةٌ. عِندما تَبدَأُ بِالتَّعَلُّمِ، سَتُلاحِظُ أَنَّ كُلَّ شَيءٍ يَتَّبِعُ نِظامًا مُحَدَّدًا، وَهذا يَجعَلُ مِنَ السَّهْلِ تَعَلُّمَ اللُّغةِ وَفَهمَ مَعاني الكَلِماتِ وَالجُمَلِ الجَميلَةِ.
"""

# Load model and tokenizer
print(f"Loading model...")
model = VitsModel.from_pretrained("facebook/mms-tts-ara")

print(f"Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")

# Tokenize text
inputs = tokenizer(text_to_convert, return_tensors="pt")

# Generate waveform
with torch.no_grad():
    output = model(**inputs).waveform

# Convert to numpy
audio_numpy = output.squeeze().cpu().numpy()  # Remove batch dim & convert to numpy

audio_save_name = "techno1.wav"

scipy.io.wavfile.write(audio_save_name, rate=model.config.sampling_rate, data=audio_numpy)
print(f"✅ Audio saved as {audio_save_name}")
