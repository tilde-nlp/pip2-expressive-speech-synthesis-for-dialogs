import io
import base64

import requests
from flask import Flask, render_template, request


# TODO: specify relevant URL
API_SYNTH = "http://localhost:8001/synthesize"
API_UPLOAD = "http://localhost:8001/upload"

app = Flask(__name__)


@app.route("/gst_tool")
def gst_tool():
    gst_tokens = {}

    for i in range(8):
        gst_tokens[f"{i}"] = 0.0

    return render_template("gst_tool.html", gst_tokens=gst_tokens)


@app.route("/bot_interface")
def bot_interface():
    return render_template("bot_interface.html")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    request_body = request.get_json()

    if request_body is None:
        return {}, 400

    text = request_body.get('text', None)
    gst_tokens = request_body.get('gst_tokens', None)
    reference_style = request_body.get("reference_style", None)
    reference_name = request_body.get("reference_name", None)
    """
    reference_style - base64 encoded audio
    reference_name - string name for previously uploaded reference audio
    """

    body = {
        "text": text
    }

    if gst_tokens is not None:
        body["gst_tokens"] = gst_tokens
    elif reference_style is not None:
        body["reference_style"] = reference_style
    elif reference_name is not None:
        body["reference_name"] = reference_name

    res = requests.post(API_SYNTH, json=body)

    buf = io.BytesIO(res.content)

    audio_base64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "audio": audio_base64
    }


@app.route("/upload", methods=["POST"])
def upload():
    request_body = request.get_json()

    if request_body is None:
        return {}, 400

    reference_style = request_body.get("reference_style", None)
    reference_name = request_body.get("reference_name", None)

    if reference_style is None or reference_name is None:
        return {}, 400

    body = {
        "reference_style": reference_style,
        "reference_name": reference_name,
    }

    res = requests.post(API_UPLOAD, json=body)
    res_json = res.json()

    if "id" not in res_json:
        return {}, 400

    return {
        "id": res_json["id"]
    }


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
