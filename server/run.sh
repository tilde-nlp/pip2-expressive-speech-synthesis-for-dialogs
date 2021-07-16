#!/bin/bash

PYTHONPATH=tacotron2_gst:$PYTHONPATH python server/main.py \
	-c server/config.yaml
