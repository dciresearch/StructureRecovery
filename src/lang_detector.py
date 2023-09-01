import fasttext
import re
from typing import List, Union
from pathlib import Path

LANGUAGE_DETECTION_MODEL_PATH = "lang_detect/lid.176.ftz"

lang_aliases = {
    "ru": "russian",
    "en": "english"
}


def get_supported_languages():
    return lang_aliases.keys()


class LanguageDetector:
    def __init__(self, model_dir="./models") -> None:
        model_dir = Path(model_dir)
        model_path = (model_dir / LANGUAGE_DETECTION_MODEL_PATH).as_posix()
        self.model = fasttext.load_model(model_path)
        self._digital_words = re.compile(r"\b\w*\d+\w*\b")
        self._spaces = re.compile(r"\s+")

    def _normalize_text(self, text: str):
        text_norm = self._digital_words.sub("", text)
        text_norm = self._spaces.sub(" ", text_norm)
        return text_norm

    def detect(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        texts = [re.sub(r"([^\.\!\?\n])\n", r"\1.\n", t) for t in texts]
        normalized_texts = [self._normalize_text(text) for text in texts]
        prediction = self.model.predict(normalized_texts)
        labels = [label[0].rsplit("__")[-1] for label in prediction[0]]
        probs = [prob[0] for prob in prediction[1]]
        return labels, probs
