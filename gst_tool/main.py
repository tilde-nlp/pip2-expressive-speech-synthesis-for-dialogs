import io
import base64

import requests
from flask import Flask, render_template, request


# TODO: specify relevant URL
URL = "http://localhost:8001/synthesize"

app = Flask(__name__)


@app.route("/")
def index():
    gst_tokens = {}

    for i in range(8):
        gst_tokens[f"{i}"] = 0.0

    return render_template("index.html", gst_tokens=gst_tokens)


@app.route("/synthesize", methods=["POST"])
def synthesize():
    request_body = request.get_json()

    if request_body is None:
        return {}, 400

    text = request_body.get('text', None)
    gst_tokens = request_body.get('gst_tokens', None)
    reference_style = request_body.get("reference_style", None)

    body = {
        "text": text
    }

    if gst_tokens is not None:
        body["gst_tokens"] = gst_tokens
    elif reference_style is not None:
        body["reference_style"] = reference_style

    res = requests.post(URL, json=body)

    buf = io.BytesIO(res.content)

    audio_base64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "audio": audio_base64
    }


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
