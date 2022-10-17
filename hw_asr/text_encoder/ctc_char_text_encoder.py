from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from tqdm.auto import tqdm
from pyctcdecode import build_ctcdecoder
import numpy as np

class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        with open('librispeech-vocab.txt') as f:
            vocab = [line.strip().lower() for line in f]

        self.decoder = build_ctcdecoder(
            [''] + self.alphabet,
            alpha=0.7,
            beta=0.1,
            kenlm_model_path='lm.arpa',
            unigrams=vocab,
        )

    def ctc_decode(self, inds: List[int]) -> str:
        res = []
        last = self.EMPTY_TOK

        for i in inds:
            ch = self.ind2char[i]
            if ch == last:
                continue

            if ch != self.EMPTY_TOK:
                res.append(ch)
            last = ch

        return ''.join(res)

    def ctc_beam_search(
        self,
        probs: torch.tensor,
        probs_length,
        beam_size: int = 100,
        result_size: int = 10,
        impl_type: str = 'custom') -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """

        assert impl_type in ['custom', 'library']
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        if impl_type == 'library':
            assert result_size == 1
            return [Hypothesis(self.decoder.decode(probs[:probs_length].cpu().numpy(), beam_width=beam_size), 1.0)]

        if impl_type == 'custom':
            def step(words, probs):
                res = dict()
                for text, last, prob in words:
                    for i, p in enumerate(probs):
                        ch = self.ind2char[i]
                        next_text = text if (ch == last or ch == self.EMPTY_TOK) else text + ch

                        if (next_text, ch) not in res:
                            res[(next_text, ch)] = 0
                        res[(next_text, ch)] += prob * p

                hyp = [(text, ch, prob) for (text, ch), prob in res.items()]
                return sorted(hyp, key=lambda x: x[2], reverse=True)[:beam_size]

            beam = [('', self.EMPTY_TOK, 1.0)]
            for p in probs[:probs_length].cpu().numpy():
                beam = step(beam, np.exp(p))

            hypos = [Hypothesis(text, prob) for text, _, prob in beam]

            return sorted(hypos, key=lambda x: x.prob, reverse=True)[:result_size]
