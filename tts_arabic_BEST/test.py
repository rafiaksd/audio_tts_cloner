# pip install git+https://github.com/nipponjo/tts_arabic.git
import time, re, winsound
import numpy as np
import soundfile as sf
from tts_arabic import tts

def get_time_elapsed(start_time):
    now_time = time.time()
    print(f"⏰ Time elapsed: {now_time-start_time:.2f}s")

def generate_arabic_text_from_audio(text, output_file_name = "output_full.wav"):
    def split_arabic_with_pauses(text, max_chars=250):
        """Split text at punctuation, return (chunk, pause_ms)"""
        punctuation_pauses = {'.':400, '؟':400, '!':400, '،':300}
        
        # Split while keeping punctuation
        parts = re.split(r'([.!؟،])', text)
        chunks = []
        buffer = ""
        last_pause = 0
        
        i = 0
        while i < len(parts):
            sentence = parts[i].strip()
            punct = parts[i+1] if i+1 < len(parts) else ""
            piece = f"{sentence}{punct}".strip()
            pause_ms = punctuation_pauses.get(punct, 200) if punct else 200
            
            if len(buffer) + len(piece) > max_chars:
                if buffer:
                    chunks.append((buffer.strip(), last_pause))
                buffer = piece
            else:
                buffer += " " + piece if buffer else piece
            
            last_pause = pause_ms
            i += 2 if punct else 1  # skip punctuation if it exists

        if buffer:
            chunks.append((buffer.strip(), last_pause))
        
        return chunks

    chunks = split_arabic_with_pauses(text)
    all_wave = []
    sr = 22050  # sampling rate

    for chunk_text, pause_ms in chunks:
        chunk_start_time = time.time()

        print(f"Generating for chunk: {chunk_text}")
        wave_chunk = tts(
            chunk_text,
            speaker=1,
            pace=0.9,
            denoise=0.005,
            volume=0.9,
            pitch_mul=1,
            pitch_add=0,
            model_id='fastpitch',
            vocoder_id='hifigan',
            cuda=None,  # None CPU, 0 GPU
            bits_per_sample=32
        )
        
        all_wave.append(wave_chunk)
        
        # Add silent pause after chunk
        silence_samples = int(sr * pause_ms / 1000)
        all_wave.append(np.zeros(silence_samples, dtype=np.float32))

        get_time_elapsed(chunk_start_time)

    final_wave = np.concatenate(all_wave)
    sf.write(output_file_name, final_wave, sr)

my_text = """
مَا يُبْتَلَى الْإِنْسَانُ إِلَّا بِسَبَبِ ذَنْبٍ، وَمَا يُرْفَعُ إِلَّا بِتَوْبَةٍ. فَأَحْيَانًا الْإِنْسَانُ، كَثِيرٌ مِنَ النَّاسِ الآنَ، عِنْدَمَا يُبْتَلَى لَا يُفَكِّرُ فِي هَذَا الْأَمْرِ. يَعْنِي بَعْضُ النَّاسِ الآنَ يُبْتَلَى بِوُضُوحٍ. طَيِّب، أَوَّلُ خُطْوَةٍ، أَوَّلُ خُطْوَةٍ: اسْأَلْ لِمَاذَا؟ قَد تَكُونُ أَنْتَ مُقَصِّرًا، أَنْتِ مُقَصِّرَةً فِي فَرْضٍ مِنْ فَرَائِضِ اللَّهِ، فِي ذَنْبٍ مِنَ الذُّنُوبِ. مِثَالًا الْإِنْسَانُ يُرَاجِعُ نَفْسَهُ، يُحَاسِبُ نَفْسَهُ. الشَّيْءُ الْآخَرُ، آخَرُ: الْحَمْدُ لِلَّهِ الَّذِي

وَعِنْدَمَا يُوَاجِهُ الْإِنْسَانُ صُعُوبَاتٍ، يَجِبُ عَلَيْهِ أَنْ يَتَذَكَّرَ أَنَّ كُلَّ مُشْكِلَةٍ تَحْمِلُ فِيهَا فُرْصَةً لِلْتَّعَلُّمِ وَالنُّموِّ. لَا يَجِبُ أَنْ يَسْتَسْلِمَ لِلْيَأْسِ وَالْإِحْبَاطِ، بَلْ عَلَيْهِ أَنْ يَبْحَثَ عَنْ الْحُلُولِ وَيُحَافِظَ عَلَى إِيمَانِهِ وَأَمَلِهِ. فَالصَّبْرُ وَالاعْتِمَادُ عَلَى اللَّهِ هُمَا السَّلِيلَةُ إِلَى تَخْطِي الصِّعَابِ وَالوُصُولِ إِلَى النَّجَاحِ.

وَأَخِيرًا، يَجِبُ عَلَى الْإِنْسَانِ أَنْ يَكُونَ شَاكِرًا لِكُلِّ نِعْمَةٍ أُعْطِيَهَا  من اللَّه، صَغِيرَةً كَانَتْ أَوْ كَبِيرَةً. الشُّكْرُ يُنَمِّي الْقَلْبَ وَيُطَهِّرُ النَّفْسَ، وَيَجْعَلُ الْإِنْسَانَ يَرَى الْحَيَاةَ بِنَظَرَةٍ إِيجَابِيَّةٍ وَرَحِيمَةٍ. فَتَقْوَى اللَّهِ وَالاعْتِرَافُ بِالنِّعَمِ هُمَا مِفْتَاحَا السَّعَادَةِ وَالاطْمِئْنَانِ الدَّاخِلِيِّ
"""

started_time = time.time()

generate_arabic_text_from_audio(my_text)

get_time_elapsed(started_time)
winsound.Beep(1000,500)