import sys
import base64
import logging
import argparse
from io import BytesIO

import yaml
from sanic import Sanic
from scipy.io.wavfile import write
from sanic.response import json, raw


app = Sanic("PIP2 Expressive TTS Server")


@app.route('/synthesize', methods=["POST"])
def synthesize(request):
    body = request.json

    text = body.get("text", None)
    gst_tokens = body.get("gst_tokens", None)
    reference_style = body.get("reference_style", None)
    speaker_id = body.get("speaker_id", 0)

    if text is None:
        return json({})

    if gst_tokens is not None:
        gst_style = {}
        for i in range(len(gst_tokens)):
            gst_style[i] = gst_tokens[i]
    elif reference_style is not None:
        audio_base64 = reference_style.split("base64,")[-1]
        audio = base64.b64decode(audio_base64)
        gst_style = BytesIO(audio)
    else:
        gst_style = {
            "0": 0.15,
            "1": -0.15,
            "5": 0.15
        }

    pcm = tts_engine.speak_with_style(text, gst_style, speaker_id)
    buf = BytesIO()
    write(buf, tts_engine.get_sample_rate(), pcm)
    buf.seek(0)

    return raw(buf.read(), content_type="audio/wav")


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
