# Expressive Latvian Speech Synthesis for Dialog Systems

This repository contains the prototype source code for our Interspeech 2021 Show & Tell paper "Expressive Latvian Speech Synthesis for Dialog Systems".

## Setup

1. Clone this repo: `git clone https://github.com/tilde-nlp/pip2-expressive-speech-synthesis-for-dialogs.git`
2. CD into this repo: `cd pip2-expressive-speech-synthesis-for-dialogs`
3. Initialize submodule: `git submodule init; git submodule update`
4. Install [PyTorch]
5. Install [Apex]
6. Install python requirements in each subdirectory 
    - `pip install -r gst_tool/requirements.txt`
    - `pip install -r server/requirements.txt`
    - `pip install -r tacotron2_gst/requirements.txt`

## Training

### From scratch

```
PYTHONPATH=tacotron2_gst:$PYTHONPATH python tacotron2_gst/train.py \
	-o outdir \
	-l logs \
	-hp hparams.yaml
```

### From pre-trained model

```
PYTHONPATH=tacotron2_gst:$PYTHONPATH python tacotron2_gst/train.py \
	-o outdir \
	-l logs \
	-hp hparams.yaml \
	-c /path/to/checkpoint \
	--warm_start  # ommit this flag to continue training from the checkpoint
```

### Dataset structure

Specify the train and validation filelists in `hparams.yaml` with the following structure:

With multi-speaker option disabled:
```
/path/to/audio1.wav|text1
/path/to/audio2.wav|text2
...
```

With multi-speaker option enabled:
```
/path/to/audio1.wav|text1|0
/path/to/audio2.wav|text2|0
/path/to/audio3.wav|text3|1
/path/to/audio4.wav|text4|1
...
```
where `|0` corresponds to the speaker id.

## Inference

```
PYTHONPATH=tacotron2_gst:$PYTHONPATH python tacotron2_gst/synth.py \
	-f lines.txt \
	-c /path/to/tacotron2_gst_checkpoint \
	-w /path/to/waveglow_checkpoint \
	-hp hparams.yaml \
	-o audio_outdir \
	-sid 0 \  # specify speaker id, if use_speaker_embedding == true
	--gst_style /path/to/wav \  # specify style reference, to use dictionary input, pass it through engine.py
	--cuda  # ommit to use CPU
```

## GST tool

To run the GST tool:
1. Start the TTS server by running `server/run.sh`
2. Start the GST tool page by running `python gst_tool/main.py`

## Acknowledgements

This research has been supported by the European Regional Development Fund within the joint project of SIA TILDE and University of Latvia “Multilingual Artificial Intelligence Based Human Computer Interaction” No. 1.1.1.1/18/A/148.

This repository uses code from the following repos:
* [NVIDIA Tacotron 2](https://github.com/NVIDIA/tacotron2)
* [NVIDIA WaveGlow](https://github.com/NVIDIA/waveglow)
* [Mozilla TTS](https://github.com/mozilla/TTS)


[pytorch]: https://github.com/pytorch/pytorch#installation
[Apex]: https://github.com/nvidia/apex