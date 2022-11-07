#coding: utf-8

import os
import time
import random
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from dataset_maker.emotion_mapping import emotion_map

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    """Dataset container

    Args:
        Dataset: extend base torch class Dataset
    """    
    
    def __init__(self,
                 dataset_path: str,
                 sr=24000,
                 validation=False,
                 sep="|"                 
                 ):
        """_summary_

        Args:
            dataset_path (str): _description_
            sr (int, optional): _description_. Defaults to 24000.
            validation (bool, optional): _description_. Defaults to False.
            sep (str, optional): _description_. Defaults to ";".

        Raises:
            FileExistsError: _description_
        """        
        logger.info(f"MelDataset:__init__: Constructor call of dataset")
        logger.info(f"MelDataset:__init__: Check existence of dataset path..")
        
        logger.info(f"MelDataset:__init__: Load dataset into a dataframe object..")
        self.dataset = pd.read_csv(dataset_path, sep=sep, names=["source_path","reference_path","reference_emotion"])
        self.dataset["already_used"] = False
         
        logger.info(f"MelDataset:__init__: Ok.")

        logger.info(f"MelDataset:__init__: Sampling Rate {sr}")
        self.sr = sr
        
        logger.info(f"MelDataset:__init__: Dataset scope -> {validation}, (False) training/ (True) validation")
        self.validation = validation
        
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        self.mean, self.std = -4, 4
        self.max_mel_length = 192

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        if self.dataset.iloc[idx]["already_used"]:
            raise IndexError("")
        row = self.dataset.iloc[idx]
        self.dataset.iloc[idx]["already_used"] = True
        mel_tensor, label = self._load_data(row["source_path"])
        ref_mel_tensor, ref_label = self._load_data(row["reference_path"],emotion_map[row["reference_emotion"]])
        # In emotion embedding the speaker is not important, instead emotion is.
        # Style diversification loss is ambiguous in our task, a way to keep it forgettable without change a lot of code could be this way
        ref2_mel_tensor = ref_mel_tensor
        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label
    
    def _load_data(self, wav_path: str, label: int = emotion_map["neutral"]):
        """Produce mel-spectrogram given a wav file

        Args:
            wav_path (str): Wav path of the source file
            label (int, optional): Label(emotion) check emotion_map. Defaults to emotion_map["neutral"].

        Returns:
            (torch.tensor, int): Mel-Spectrogram of the wav file, label 
        """        
        wave_tensor = self._generate_wav_tensor(wav_path)
        
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        # HP. source and reference audio, since the statement is the same, so same phoneme, the duration will be the same.
        # The cut will happen for source and reference
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, label

    def _preprocess(self, wave_tensor):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _generate_wav_tensor(self, wave_path: str) -> torch.tensor:
        """Private methods that trasform a wav file into a tensor

        Args:
            wave_path (str): path of the source wav file

        Returns:
            torch.tensor: tensorial representation of source wav
        """        
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()

        for bid, (mel, label, ref_mel, ref2_mel, ref_label) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label

        z_trg = torch.randn(batch_size, self.latent_dim)
        # In emotion embedding the speaker is not important, instead emotion is.
        # Style diversification loss is ambiguous in our task, a way to keep it forgettable without change a lot of code could be this way
        z_trg2 = z_trg
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
