import logging
import numpy as np
import torch as th
import random
from librosa.util import find_files
from torch.utils import data
from torch.utils.data import DataLoader



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class TasDataset(object):
    def __init__(self, file_path):
        self.filelist = find_files(file_path, "npz")

    def __getitem__(self, idx):
        dat = np.load(self.filelist[idx])
        mix_speech = dat["mix_speech"]
        speech1 = dat["speech1"]
        speech2 = dat["speech2"]
        return th.from_numpy(mix_speech), th.from_numpy(speech1), th.from_numpy(speech2)

    def __len__(self):
        return len(self.filelist)

def test():
    train_dataset = TasDataset('./data/train_input/')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)
    i = 0
    for mix_speech, speech1, speech2 in train_loader:
        while i<10:
            print(mix_speech.shape)
            print(speech1.shape)
            print(speech2.shape)
            print('\n')
            i += 1



if __name__ == '__main__':
    test()



