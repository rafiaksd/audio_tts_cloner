import outetts

# Initialize the interface
print(f"\n\n============== 1 INTERFACING STARTING")
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,  # points to INT4
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.Q4_K_M  # important!
    )
)

# Load the default speaker profile
# speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

# Or create your own speaker profiles in seconds and reuse them instantly
print(f"\n\n============== 2 CREATING SPEAKER")
speaker = interface.create_speaker("some.wav")
interface.save_speaker(speaker, "speaker.json")
speaker = interface.load_speaker("speaker.json")

# Generate speech
print(f"\n\n============== 3 GENERATING AUDIO")
output = interface.generate(
    config=outetts.GenerationConfig(
        text="ما يبتلى الإنسان إلا بسبب ذنب وما يرفع إلا بتوبة فأحيانًا الإنسان كثير من الناس الآن عندما يبتلى ما يفكر في هذا الأمر يعني بعض الناس الآن يبتلى بوضوح طيب أول خطوة أول خطوة اسأل لماذا قد تكون أنت مقصر أنت مقصرة في فرد من فرائض الله في ذنب من الذنوب مثلا الإنسان يراجع نفسه يحاسب نفسه الشيء الآخر آخر الحمد لله اللي",
        speaker=speaker,
    )
)

print(f"\n\n============== 4 AUDIO SAVING")
output_file_name = "output outetts.wav"
output.save(output_file_name)