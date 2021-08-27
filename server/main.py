import os
import sys
import json
import uuid
import base64
import logging
import argparse
from io import BytesIO

import yaml
from sanic import Sanic, response
from scipy.io.wavfile import write


app = Sanic("PIP2 Expressive TTS Server")


REF_DIR = "data"


@app.route('/synthesize', methods=["POST"])
def synthesize(request):
    body = request.json

    text = body.get("text", None)
    gst_tokens = body.get("gst_tokens", None)
    reference_style = body.get("reference_style", None)
    reference_name = body.get("reference_name", None)
    speaker_id = body.get("speaker_id", 1)

    if text is None:
        return response.json({}, status=400)

    if gst_tokens is not None:
        gst_style = gst_tokens
    elif reference_style is not None:
        audio_base64 = reference_style.split("base64,")[-1]
        audio = base64.b64decode(audio_base64)
        gst_style = BytesIO(audio)
    elif reference_name is not None:
        if os.path.isfile(os.path.join(REF_DIR, "mapping.json")):
            with open(os.path.join(REF_DIR, "mapping.json"), "r", encoding="utf-8") as f:
                mapping = json.load(f)
                if reference_name not in mapping:
                    return response.json({"error", "Unknown reference name"}, status=400)

                gst_style = mapping[reference_name]
        else:
            return response.json({"error", "No reference audio saved"}, status=400)
    else:
        gst_style = None

    pcm = tts_engine.speak_with_style(text, gst_style, speaker_id)
    buf = BytesIO()
    write(buf, tts_engine.get_sample_rate(), pcm)
    buf.seek(0)

    return response.raw(buf.read(), content_type="audio/wav")


@app.route("/upload", methods=["POST"])
def upload(request):
    body = request.json

    reference_style = body.get("reference_style", None)
    reference_name = body.get("reference_name", None)
    if reference_style is None or reference_name is None:
        return response.json({}, status=400)

    audio_base64 = reference_style.split("base64,")[-1]
    audio = base64.b64decode(audio_base64)

    # save audio
    ref_id = str(uuid.uuid4())
    ref_path = os.path.join(REF_DIR, f"{ref_id}.wav")
    with open(ref_path, "wb") as f:
        f.write(audio)

    # save mapping from ref name to audio
    if os.path.isfile(os.path.join(REF_DIR, "mapping.json")):
        with open(os.path.join(REF_DIR, "mapping.json"), "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = {}

    mapping[reference_name] = ref_path

    with open(os.path.join(REF_DIR, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    return response.json({
        "id": ref_id
    })


def create_tts_engine(config_path: str):
    logging.info("Initializing TTS Engine...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sys.path.append(config["tacotron2_gst_path"])
    sys.path.append(config["waveglow_path"])
    from tacotron2_gst.engine import TTSEngine

    tts_engine = TTSEngine()
    logging.info("Loading model...")
    tts_engine.load_model(config["model_checkpoint_path"],
                          config["vocoder_checkpoint_path"],
                          config["model_config_path"],
                          config["synth_device"])

    logging.info("TTS Engine initialized")
    return tts_engine


def main(args):
    logging.getLogger().setLevel(logging.DEBUG)

    global tts_engine
    tts_engine = create_tts_engine(args.config_path)

    app.run(host="0.0.0.0", port=8001, debug=True, auto_reload=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=None, required=True,
                        help="Path to server config.")
    args = parser.parse_args()

    main(args)
