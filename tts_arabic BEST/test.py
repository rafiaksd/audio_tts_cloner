#!pip install git+https://github.com/nipponjo/tts_arabic.git
 
import time
from tts_arabic import tts

def get_time_elapsed(start_time):
     now_time = time.time()
     print (f"⏰ Time elapsed: {now_time-start_time:.2f}s")

text = """
مَا يُبْتَلَى الْإِنْسَانُ إِلَّا بِسَبَبِ ذَنْبٍ، وَمَا يُرْفَعُ إِلَّا بِتَوْبَةٍ. فَأَحْيَانًا الْإِنْسَانُ، كَثِيرٌ مِنَ النَّاسِ الآنَ، عِنْدَمَا يُبْتَلَى لَا يُفَكِّرُ فِي هَذَا الْأَمْرِ. يَعْنِي بَعْضُ النَّاسِ الآنَ يُبْتَلَى بِوُضُوحٍ. طَيِّب، أَوَّلُ خُطْوَةٍ، أَوَّلُ خُطْوَةٍ: اسْأَلْ لِمَاذَا؟ قَد تَكُونُ أَنْتَ مُقَصِّرًا، أَنْتِ مُقَصِّرَةً فِي فَرْضٍ مِنْ فَرَائِضِ اللَّهِ، فِي ذَنْبٍ مِنَ الذُّنُوبِ. مِثَالًا الْإِنْسَانُ يُرَاجِعُ نَفْسَهُ، يُحَاسِبُ نَفْسَهُ. الشَّيْءُ الْآخَرُ، آخَرُ: الْحَمْدُ لِلَّهِ الَّذِي

وَعِنْدَمَا يُوَاجِهُ الْإِنْسَانُ صُعُوبَاتٍ، يَجِبُ عَلَيْهِ أَنْ يَتَذَكَّرَ أَنَّ كُلَّ مُشْكِلَةٍ تَحْمِلُ فِيهَا فُرْصَةً لِلْتَّعَلُّمِ وَالنُّموِّ. لَا يَجِبُ أَنْ يَسْتَسْلِمَ لِلْيَأْسِ وَالْإِحْبَاطِ، بَلْ عَلَيْهِ أَنْ يَبْحَثَ عَنْ الْحُلُولِ وَيُحَافِظَ عَلَى إِيمَانِهِ وَأَمَلِهِ. فَالصَّبْرُ وَالاعْتِمَادُ عَلَى اللَّهِ هُمَا السَّلِيلَةُ إِلَى تَخْطِي الصِّعَابِ وَالوُصُولِ إِلَى النَّجَاحِ.

وَأَخِيرًا، يَجِبُ عَلَى الْإِنْسَانِ أَنْ يَكُونَ شَاكِرًا لِكُلِّ نِعْمَةٍ أُعْطِيَهَا اللَّهُ، صَغِيرَةً كَانَتْ أَوْ كَبِيرَةً. الشُّكْرُ يُنَمِّي الْقَلْبَ وَيُطَهِّرُ النَّفْسَ، وَيَجْعَلُ الْإِنْسَانَ يَرَى الْحَيَاةَ بِنَظَرَةٍ إِيجَابِيَّةٍ وَرَحِيمَةٍ. فَتَقْوَى اللَّهِ وَالاعْتِرَافُ بِالنِّعَمِ هُمَا مِفْتَاحَا السَّعَادَةِ وَالاطْمِئْنَانِ الدَّاخِلِيِّ
"""

started_time = time.time()

wave = tts(
    text,
    speaker=1,        # speaker ID: 0, 1, 2, or 3 :contentReference[oaicite:1]{index=1}  
    pace=0.9,          # speed / pace of speech :contentReference[oaicite:2]{index=2}  
    denoise=0.005,     # reduce vocoder noise :contentReference[oaicite:3]{index=3}  
    volume=0.9,        # volume of output :contentReference[oaicite:4]{index=4}  
    pitch_mul=1,       # pitch multiplier :contentReference[oaicite:5]{index=5}  
    pitch_add=0,       # pitch offset :contentReference[oaicite:6]{index=6}  
    model_id='fastpitch',   # use FastPitch model :contentReference[oaicite:7]{index=7}  
    vocoder_id='hifigan',   # choose a vocoder: hifigan or vocos :contentReference[oaicite:8]{index=8}  
    cuda=None,         # set GPU device if using CUDA :contentReference[oaicite:9]{index=9}  
    save_to="output1long.wav",   # path to save WAV file :contentReference[oaicite:10]{index=10}  
    bits_per_sample=32     # bit depth for saving :contentReference[oaicite:11]{index=11}  
)

# wave is a numpy array containing the waveform

get_time_elapsed(started_time)