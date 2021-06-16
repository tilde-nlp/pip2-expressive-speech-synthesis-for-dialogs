a = document.createElement("a");
refAudio = document.createElement("audio");
synthAudio = document.createElement("audio");

$(() => {
    $("#synth-button").click(() => synthesize());
    $("#copy-button").click(() => {
        let tokenValues = $(".token-slider").map((idx, elem) => {
            return $(elem).val();
        }).get();
        tokenValues = tokenValues.map((val) => (parseInt(val) / 10));

        let output = {};
        for (let i = 0; i < tokenValues.length; i++) {
            output[i] = tokenValues[i];
        }
    });

    $("#file-close-button").click(() => {
        $("#style-reference-playback-container").get(0).style.display = "none";

        let styleInput = $("#style-reference-input").get(0);
        styleInput.value = styleInput.defaultValue;
        $("#style-reference-button").get(0).style.display = "block";
        $(".token-slider").prop("disabled", false);
    });

    $("#file-play-button").click(async () => {
        const file = $("#style-reference-input").get(0).files[0];
        const b64 = await blobToData(file);
        refAudio.setAttribute( "src", b64);
        refAudio.play();
    });

    $("#download-button").click(() => {
        // TODO
    });

    $("#style-reference-button").click(() => $("#style-reference-input").click());

    $("#style-reference-input").change(() => {
        $("#style-reference-playback-container").get(0).style.display = "flex";
        $("#style-reference-button").get(0).style.display = "none";
        const filename = $("#style-reference-input").val().split('\\').pop();
        $("#style-reference-filename").text(filename);
        $(".token-slider").prop("disabled", true);
    });
});

last_audio = null;

const blobToData = (blob) => {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result)
    reader.readAsDataURL(blob)
  })
}

const synthesize = async () => {
    const text = $("#synth-textarea").val();

    let body = { text };

    // Check if file uploaded, otherwise, use style token values
    const files = $("#style-reference-input").get(0).files;

    if (files.length !== 0) {
        let file = files[0];
        body.reference_style = await blobToData(file);
    } else {
        let tokenValues = $(".token-slider").map((idx, elem) => {
            return $(elem).val();
        }).get();
        body.gst_tokens = tokenValues.map((val) => (parseInt(val) / 10));
    }

    let req = $.ajax({
        type: "POST",
        url: "/synthesize",
        data: JSON.stringify(body),
        dataType: "json",
        contentType: "application/json;charset=utf-8"
    });

    req.done((res) => {
        synthAudio.setAttribute( "src", "data:audio/wav;base64," + res.audio);
        synthAudio.controls = true
        synthAudio.play();
    });
}
