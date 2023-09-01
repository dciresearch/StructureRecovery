from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.signal import argrelextrema
from itertools import chain, tee, islice
from pathlib import Path
from razdel import sentenize


def np_norm(a):
    return a/np.linalg.norm(a, axis=1, keepdims=True)


def np_cos_sim(a, b):
    a = np_norm(a)
    b = np_norm(b)
    return a@b.T


def smooth(x, window_size=2):
    s = np.r_[2 * x[0] - x[window_size:1:-1],
              x, 2 * x[-1] - x[-1:- window_size:-1]]
    kernel = np.ones(window_size, "d")

    y = np.convolve(kernel / kernel.sum(), s, mode="same")
    return y[window_size - 1: - window_size + 1]


def russian_sentenize(text):
    return [s.text for s in sentenize(text)]


model_zoo = {
    "ru": "encoders/paraphrase-multilingual-MiniLM-L12-v2",
    "en": "encoders/paraphrase-multilingual-MiniLM-L12-v2",
}

sentenizer_zoo = {
    "ru": russian_sentenize,
}


def nwise(iterable, n=2):
    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return zip(*iters)


class Paragraphizer:
    def __init__(self, language="ru", sentenizer=None, model_dir="./models"):
        model_dir = Path(model_dir)
        model_path = model_zoo.get(language, None)
        if model_path is None:
            raise ValueError(f"Language {language} is not supported")
        self.encoder = SentenceTransformer((model_dir / model_path).as_posix())
        if sentenizer is None:
            sentenizer = sentenizer_zoo.get(language, None)
        self.sentenizer = sentenizer

    def breakdown(self, text):
        sentences = self.sentenizer(text)
        sent_embs = self.encoder.encode(sentences)
        par_breaks = self.find_break_points(sent_embs, smoothed=False)
        par_splits = self.splits_from_breakpoints(par_breaks)
        pars = [" ".join(sentences[sl]) for sl in par_splits]
        return pars

    @staticmethod
    def find_break_points(sent_embs, smoothed=False):
        sims = np_cos_sim(sent_embs, sent_embs)
        bi_sims = np.diag(sims, 1)
        if smoothed:
            bi_sims = smooth(bi_sims)
        par_breaks = argrelextrema(bi_sims, np.less)[0].tolist()
        return par_breaks

    @staticmethod
    def splits_from_breakpoints(breakpoints):
        return [slice(*sl) for sl in nwise(chain([0], breakpoints, [None]))]
