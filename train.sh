CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tacotron2_gst:$PYTHONPATH python tacotron2_gst/train.py \
	-o outdir \
	-l logs \
	-hp hparams.yaml
