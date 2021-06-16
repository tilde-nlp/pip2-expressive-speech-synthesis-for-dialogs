"""
Adapted from https://github.com/NVIDIA/tacotron2

Cleaners are transformations that run over the input text at both training and eval time.
Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners" hyperparameter.
"""
import re


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def lv_cleaners(text):
    text = lowercase(text)
    text = text.replace("\n", " ")
    text = text.strip()
    text = collapse_whitespace(text)
    return text


def basic_cleaners(text):
    """
    Basic pipeline that lowercases and collapses whitespace without transliteration.
    """
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
