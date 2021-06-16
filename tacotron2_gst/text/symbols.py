"""
Adapted from https://github.com/NVIDIA/tacotron2

Defines the set of symbols used in text input to the model.
"""
pad = '_'
bos = '^'  # beginning of sequence
eos = '$'  # end of sequence
_punctuation = '.,;:!?-\'"()/\\ '
_letters = 'aābcčdeēfgģhiījkķlļmnņopqrsštuūvwxyzž'

# Export all symbols:
symbols = [pad, bos, eos] + list(_punctuation) + list(_letters)
