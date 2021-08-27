a = document.createElement("a");
refAudio = document.createElement("audio");
synthAudio = document.createElement("audio");

$(() => {
    $("#synth-button").click(() => synthesize());
    $("#copy-button").click(() => copyWeights());
    $("#download-button").click(() => download_audio(last_audio));
    $("#file-close-button").click(() => onReferenceRemoved());
    $("#file-play-button").click(() => play());
    $("#style-reference-button").click(() => $("#style-reference-input").click());
    $("#style-reference-upload").click(() => uploadReference());
    $("#style-reference-input").change(() => onReferenceSelected());
});

last_audio = null;

const blobToData = (blob) => {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result)
    reader.readAsDataURL(blob)
  })
}

const play = async () => {
    const file = $("#style-reference-input").get(0).files[0];
    const b64 = await blobToData(file);
    refAudio.setAttribute( "src", b64);
    refAudio.play();
}

const onReferenceSelected = () => {
    $("#reference-select-form").get(0).style.display = "none";
    $("#reference-interaction-form").get(0).style.display = "flex";

    $("#file-id-container").get(0).style.display = "block";

    const filename = $("#style-reference-input").val().split('\\').pop();
    $("#style-reference-filename").text(filename);
    $(".token-slider").prop("disabled", true);
    $("#copy-button").prop("disabled", true);
}

const onReferenceRemoved = () => {
    $("#reference-select-form").get(0).style.display = "block";
    $("#reference-interaction-form").get(0).style.display = "none";

    $("#file-id-container").get(0).style.display = "none";
    $("#file-id-container").text("");

    let styleInput = $("#style-reference-input").get(0);
    styleInput.value = styleInput.defaultValue;

    $(".token-slider").prop("disabled", false);
    $("#copy-button").prop("disabled", false);
    $("#style-reference-upload").prop("disabled", false);
}

const copyWeights = () => {
    let tokenValues = $(".token-slider").map((idx, elem) => {
        return $(elem).val();
    }).get();

    tokenValues = tokenValues.map((val) => (parseInt(val) / 10));

    let output = {};
    for (let i = 0; i < tokenValues.length; i++) {
        if (tokenValues[i] !== 0) {
            output[i] = tokenValues[i];
        }
    }

    copy_to_clipboard(JSON.stringify(output));
}

const uploadReference = async () => {
    const files = $("#style-reference-input").get(0).files;
    if (files.length === 0) {
        return;
    }

    let body = {
        "reference_style": await blobToData(files[0]),
        "reference_name": $("#reference-name").val()
    };

    let req = $.ajax({
        type: "POST",
        url: "/upload",
        data: JSON.stringify(body),
        dataType: "json",
        contentType: "application/json;charset=utf-8"
    });

    req.done((res) => {
        if (res && res.id) {
            $("#style-reference-upload").prop("disabled", true);
            $("#file-id-container").text("File uploaded");
        }
    });
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

        tokenValues = tokenValues.map((val) => (parseInt(val) / 10));

        let output = {};
        for (let i = 0; i < tokenValues.length; i++) {
            if (tokenValues[i] !== 0) {
                output[i] = tokenValues[i];
            }
        }

        body.gst_tokens = output;
    }

    let req = $.ajax({
        type: "POST",
        url: "/synthesize",
        data: JSON.stringify(body),
        dataType: "json",
        contentType: "application/json;charset=utf-8"
    });

    req.done((res) => {
        last_audio = "data:audio/wav;base64," + res.audio;
        synthAudio.setAttribute( "src", "data:audio/wav;base64," + res.audio);
        synthAudio.controls = true
        synthAudio.play();
    });
}

const download_audio = (data) => {
    let a = document.createElement("a");
    document.body.appendChild(a);
    a.style = "display: none";
    a.href = data;
    a.download = 'speech.wav';
    a.click();
    document.body.removeChild(a);
}

const copy_to_clipboard = (text) => {
    // navigator clipboard api needs a secure context (https)
    if (navigator.clipboard && window.isSecureContext) {
        // navigator clipboard api method'
        return navigator.clipboard.writeText(text);
    } else {
        // text area method
        let textArea = document.createElement("textarea");
        textArea.value = text;
        // make the textarea out of viewport
        textArea.style.position = "fixed";
        textArea.style.left = "-999999px";
        textArea.style.top = "-999999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        return new Promise((res, rej) => {
            // here the magic happens
            document.execCommand('copy') ? res() : rej();
            textArea.remove();
        });
    }
}
