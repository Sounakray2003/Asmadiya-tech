from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="""Emma: Hey, you won’t believe what just happened!
Ryan: Oh no, don’t tell me you set the kitchen on fire again.
Emma: What?! That was one time — and it was barely a fire!
Ryan: The fire department disagreed. Anyway, what’s the big news?
Emma: I just found an envelope under my doorstep this morning… no name, no address, just a golden seal.
Ryan: That sounds like the start of a mystery movie. Did you open it?
Emma: Of course I did! Inside was a plane ticket to Paris — first class!
Ryan: Wait… what? Who sends you a plane ticket out of nowhere?
Emma: I don’t know! But there was also a note that said, “You’ve been chosen. Don’t be late.”
Ryan: Okay, either you’re being pranked… or you’re about to join some secret international adventure.
Emma: Honestly? I’m packing already.
Ryan: If you meet spies, aliens, or billionaires, call me first. I want in.""",
                file_path="output_conqui_tts/joyfull_output.wav",
                speaker_wav="Reference/joyfully.wav",
                language="en")
