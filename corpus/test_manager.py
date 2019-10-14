from unittest import TestCase

from torch.utils.data import DataLoader

from corpus import WLPDataset

import config as cfg


class TestManager(TestCase):
    def test_pytorch_dataloader(self):
        embedding_matrix, word_index, char_index = WLPDataset.prepare_embeddings(replace_digit=cfg.REPLACE_DIGITS)
        dataset = WLPDataset(word_index, char_index)

        def collate(batch):
            return batch

        data = DataLoader(dataset, batch_size=1, num_workers=8, collate_fn=collate)

        for i, item in enumerate(data):
            if i > 10:
                break
            item = item[0]
            print(dataset.to_words(item.X), item.Y, item.P, item.C, item.F)


