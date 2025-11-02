# call_xtts_triton.py
import base64
import numpy as np
import soundfile as sf
import argparse
import os
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

def to_string_numpy(s):
    return np.array([s], dtype=object)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000", help="triton server URL")
    ap.add_argument("--model", default="xtts_python")
    ap.add_argument("--text", default="In a quiet village, an old clockmaker named Arjun.")
    ap.add_argument("--speaker", default=None, help="optional path to speaker WAV for cloning")
    ap.add_argument("--out", default="out_from_triton.wav")
    args = ap.parse_args()

    client = httpclient.InferenceServerClient(url=args.url, verbose=False)

    # Prepare inputs
    input0 = httpclient.InferInput("TEXT", [1], "TYPE_STRING")
    input0.set_data_from_numpy(to_string_numpy(args.text))

    if args.speaker:
        with open(args.speaker, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("utf-8")
    else:
        b64 = ""

    input1 = httpclient.InferInput("SPEAKER_WAV_B64", [1], "TYPE_STRING")
    input1.set_data_from_numpy(to_string_numpy(b64))

    output0 = httpclient.InferRequestedOutput("AUDIO_B64")

    try:
        resp = client.infer(args.model, inputs=[input0, input1], outputs=[output0])
        audio_b64 = resp.as_numpy("AUDIO_B64")[0]
        # audio_b64 may be bytes or numpy object containing bytes
        if isinstance(audio_b64, (bytes, bytearray)):
            audio_bytes = base64.b64decode(audio_b64)
        else:
            audio_bytes = base64.b64decode(audio_b64.item())

        with open(args.out, "wb") as f:
            f.write(audio_bytes)
        print("Saved:", args.out)
    except InferenceServerException as e:
        print("Triton inference error:", e)

if __name__ == "__main__":
    main()
