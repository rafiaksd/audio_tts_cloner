import outetts

# Initialize the interface
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        # For llama.cpp backend
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16
        # For transformers backend
        # backend=outetts.Backend.HF,
    )
)

# Load the default speaker profile
# speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

# Or create your own speaker profiles in seconds and reuse them instantly
speaker = interface.create_speaker("some.wav")
interface.save_speaker(speaker, "speaker.json")
speaker = interface.load_speaker("speaker.json")

# Generate speech
output = interface.generate(
    config=outetts.GenerationConfig(
        text="ما يبتلى الإنسان إلا بسبب ذنب وما يرفع إلا بتوبة فأحيانًا الإنسان كثير من الناس الآن عندما يبتلى ما يفكر في هذا الأمر يعني بعض الناس الآن يبتلى بوضوح طيب أول خطوة أول خطوة اسأل لماذا قد تكون أنت مقصر أنت مقصرة في فرد من فرائض الله في ذنب من الذنوب مثلا الإنسان يراجع نفسه يحاسب نفسه الشيء الآخر آخر الحمد لله اللي",
        speaker=speaker,
    )
)

# Save to file
output_file_name = "output outetts.wav"
output.save(output_file_name)