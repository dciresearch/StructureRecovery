import re
import torch
from .lang_detector import LanguageDetector
from pathlib import Path

MODEL_PATH = "punct/v2_4lang_q.pt"
SUPPORTED_LANGUAGES = {'en', 'de', 'es', 'ru'}


non_letter = re.compile(r"[\W_]", flags=re.UNICODE)


def fast_unmultispace(text):
    return " ".join(text.split())


def strip_of_chars(text):
    return fast_unmultispace(non_letter.sub(" ", text))


class Punctuazier:
    def __init__(self, model_dir="./models") -> None:
        model_dir = Path(model_dir)
        self.model = torch.package.PackageImporter(
            model_dir / MODEL_PATH).load_pickle("te_model", "model")
        self.lang_det = LanguageDetector()

    @staticmethod
    def normalize(text):
        return strip_of_chars(text.lower())

    def punctuaze(self, text, len_limit=150, lang=None):
        # by implementation lang parameter just adds additional punctuation symbols for Spanish
        # normalize for model
        text = self.normalize(text)
        if lang is None:
            lang = self.lang_det.detect(text)[0][0]
        return self.model.enhance_text(text, len_limit=len_limit, lan=lang)
