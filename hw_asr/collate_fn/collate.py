import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = dict()
    try:
        for k in ['spectrogram']:
            result_batch[k] = torch.nn.utils.rnn.pad_sequence(
                [torch.squeeze(s[k]).T for s in dataset_items],
                batch_first=True,
            ).permute(0, 2, 1,)

        for k in ['audio', 'text_encoded']:
            result_batch[k] = torch.nn.utils.rnn.pad_sequence(
                [torch.squeeze(s[k]) for s in dataset_items],
                batch_first=True,
            )

        for k in ['text', 'audio_path']:
            result_batch[k] = [s[k] for s in dataset_items]

        result_batch['text_encoded_length'] = torch.tensor([s['text_encoded'].shape[1] for s in dataset_items])
        result_batch['spectrogram_length'] = torch.tensor([s['spectrogram'].shape[2] for s in dataset_items])
    except:
        result_batch['error'] = True
        print('Error in collate function!',)

    return result_batch
