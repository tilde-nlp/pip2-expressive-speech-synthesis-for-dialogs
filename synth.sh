CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tacotron2_gst:$PYTHONPATH python tacotron2_gst/synth.py \
	-f test.txt \
	-c /path/to/checkpoint \
	-w /path/to/waveglow_checkpoint \
	-hp hparams.yaml \
	-o audio_outdir \
	-sid 0 \
	--gst_style /path/to/audio.wav \
	--cuda
