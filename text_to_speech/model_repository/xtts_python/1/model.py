# model_repository/xtts_python/model.py-sdk
import os
import io
import tempfile
import base64
import traceback

import numpy as np
import soundfile as sf
import torch

from huggingface_hub import snapshot_download
from TTS.api import TTS

# Triton Python Backend utilities: available inside Triton Python backend image
from triton_python_backend_utils import TritonPythonModel, InferenceResponse, Tensor

class TritonPythonModel:
    """
    Triton Python model that wraps Coqui TTS (XTTS).
    Inputs:
      - TEXT (TYPE_STRING) : the text to synthesize
      - SPEAKER_WAV_B64 (TYPE_STRING) : optional base64-encoded WAV bytes (single item) for voice cloning
    Outputs:
      - AUDIO_B64 (TYPE_STRING) : base64-encoded WAV bytes (single item)
    """

    def initialize(self, args):
        print("[xtts_python] initialize called")
        model_config = args.get('model_config', {})
        self.model_name = model_config.get('name', 'xtts_python')
        # device selection
        self.device = "cpu" 
        print(f"[xtts_python] device => {self.device}")

        # Which huggingface model id to use (can override at runtime via env XTTS_MODEL_ID)
        self.hf_model_id = os.environ.get("XTTS_MODEL_ID", "tts_models/multilingual/multi-dataset/xtts_v2")
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)

        # Where to download the HF snapshot (inside container). Use /tmp or /models_cached
        self.download_dir = os.environ.get("XTTS_DOWNLOAD_DIR", "/tmp/hf_models")
        os.makedirs(self.download_dir, exist_ok=True)

        print(f"[xtts_python] snapshot_download model: {self.hf_model_id} -> {self.download_dir}")
        try:
            # snapshot_download returns local repo dir
            self.repo_dir = snapshot_download(repo_id=self.hf_model_id, token=hf_token, cache_dir=self.download_dir, allow_patterns=["*"])
            print(f"[xtts_python] downloaded -> {self.repo_dir}")
        except Exception as e:
            print("[xtts_python] ERROR downloading model:", e)
            raise

        # Load TTS model (Coqui TTS). TTS will load from local repo_dir if given.
        print("[xtts_python] loading TTS model from repo_dir...")
        try:
            self.tts = TTS(self.repo_dir, gpu=(self.device == "cpu"))
            print("[xtts_python] TTS model loaded.")
        except Exception:
            print("[xtts_python] Failed to load TTS model; traceback:")
            traceback.print_exc()
            raise

        # sample rate
        self.sample_rate = getattr(self.tts, "output_sample_rate", 22050)
        print(f"[xtts_python] using sample_rate: {self.sample_rate}")

    def execute(self, requests):
        """
        Execute is called with a list of InferenceRequest objects (requests argument).
        We return one InferenceResponse per request.
        """
        responses = []

        for request in requests:
            try:
                # Read TEXT input (string tensor)
                text_tensor = request.get_input_tensor_by_name("TEXT")
                text_np = text_tensor.as_numpy()
                # numpy object array of bytes, take first element
                text = text_np[0].decode("utf-8") if text_np.size > 0 else ""
                print("[xtts_python] request text:", text[:80])

                # Read SPEAKER_WAV_B64 if present
                speaker_b64 = ""
                try:
                    speaker_tensor = request.get_input_tensor_by_name("SPEAKER_WAV_B64")
                    speaker_np = speaker_tensor.as_numpy()
                    if speaker_np.size > 0:
                        # decode bytes -> str
                        speaker_b64 = speaker_np[0].decode("utf-8")
                except Exception:
                    # optional input may be missing, ignore
                    speaker_b64 = ""

                speaker_wav_path = None
                if speaker_b64:
                    try:
                        wav_bytes = base64.b64decode(speaker_b64)
                        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                        os.close(fd)
                        with open(tmp_path, "wb") as f:
                            f.write(wav_bytes)
                        speaker_wav_path = tmp_path
                        print("[xtts_python] decoded speaker wav to", speaker_wav_path)
                    except Exception:
                        traceback.print_exc()
                        speaker_wav_path = None

                # run TTS (synchronous)
                kwargs = {}
                if speaker_wav_path:
                    kwargs["speaker_wav"] = speaker_wav_path
                    # You can add speed/pitch here if allowed by the model
                    kwargs["speed"] = 0.85

                wav = self.tts.tts(text=text, **kwargs)  # numpy array float32 (-1..1)
                wav = np.asarray(wav, dtype=np.float32)

                # write wav to bytes (in-memory)
                bio = io.BytesIO()
                sf.write(bio, wav, self.sample_rate, format="WAV")
                audio_bytes = bio.getvalue()
                audio_b64 = base64.b64encode(audio_bytes)

                # Triton expects a numpy object array for TYPE_STRING outputs
                out_np = np.asarray([audio_b64], dtype=object)
                out_tensor = Tensor("AUDIO_B64", out_np)

                inference_response = InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)

            except Exception as e:
                tb = traceback.format_exc()
                print("[xtts_python] inference exception:", tb)
                # attach error to response
                inference_response = InferenceResponse(output_tensors=[], error=str(e))
                responses.append(inference_response)
            finally:
                # cleanup speaker tmp
                if 'speaker_wav_path' in locals() and speaker_wav_path and os.path.exists(speaker_wav_path):
                    try:
                        os.remove(speaker_wav_path)
                    except Exception:
                        pass

        return responses

    def finalize(self):
        print("[xtts_python] finalize called")
