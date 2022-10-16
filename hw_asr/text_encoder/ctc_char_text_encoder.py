from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


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

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        def step(words, probs):
            res = dict()
            for text, last, prob in words:
                for i, p in enumerate(probs):
                    ch = self.char2ind[i]
                    next_text = text if (ch == last or last == self.EMPTY_TOK) else text + ch

                    if (next_text, ch) not in res:
                        res[(next_text, ch)] = 0
                    res[(next_text, ch)] += prob * p

            hyp = [Hypotesis(text, prob) for (text, _), prob in res.items()]
            return sorted(hyp, key=lambda x: x.prob, reverse=True)[:beam_size]

        beam = [('', self.EMPTY_TOK, 1.0)]
        for p in probs[:probs_length]:
            beam = step(beam, p)

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
