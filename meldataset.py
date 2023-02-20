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

from typing import Tuple, Dict, List

from dataset.emotion_mapping_sad import emotion_map

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
                 dataset: pd.DataFrame,
                 validation: bool = False   
                 ):
        """Constructor

        Args:
            dataset (pd.DataFrame): Data.
            validation (bool, optional): If the dataset is in Validation mode. Defaults to False.
        """      
        self.dataset = dataset
        self.dataset["already_used"] = False
        self.validation = validation
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        self.mean, self.std = -4, 4
        self.max_mel_length = 192

    def __len__(self) -> int:
        """Cardinality of the dataset

        Returns:
            (int): The cardinality
        """        
        return self.dataset.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a sample from the set

        Args:
            idx (int): Index of the selected sample

        Raises:
            IndexError: This sample was already used 

        Returns:
            (
               (MelBand, T_Mel),
               int,
               (MelBand, T_Mel),
               (MelBand, T_Mel),
               int 
            ): (Source Spectrogram, Source label, Reference Spectrogram, Reference2 Spectrogram, Reference Label)
        """  
        if self.dataset.iloc[idx]["already_used"]:
            raise IndexError("")
        row = self.dataset.iloc[idx]
        emotion = row["reference_emotion"]
        self.dataset.iloc[idx]["already_used"] = True
        mel_tensor, label = self._load_data(row["source_path"], row["source_emotion"])
        ref_mel_tensor, ref_label = self._load_data(row["reference_path"],row["reference_emotion"])
        
        ref2 = self.dataset[self.dataset["reference_emotion"] == emotion].sample(n=1).iloc[0]
        ref2_mel_tensor, ref2_mel_label = self._load_data(ref2["reference_path"])
        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label
    
    def _load_data(self, wav_path: str, label: int = emotion_map["neutral"]) -> Tuple[torch.Tensor, int]:
        """Produce mel-spectrogram given a wav file

        Args:
            wav_path (str): Wav path of the source file
            label (int, optional): Label(emotion) check emotion_map. Defaults to emotion_map["neutral"].

        Returns:
            ((MelBand, T_Mel), int): Mel-Spectrogram of the wav file, label 
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

    def _preprocess(self, wave_tensor: torch.Tensor) -> torch.Tensor:
        """Convert to Mel-Spectrogram

        Args:
            wave_tensor (sample,1): Waveform

        Returns:
            (MelBand, T_Mel): Mel-Spectrogram of the waveform
        """
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _generate_wav_tensor(self, wave_path: str) -> torch.Tensor:
        """Private methods that trasform a wav file into a tensor

        Args:
            wave_path (str): path of the source wav file

        Returns:
            (samples,1): tensorial representation of source wav
        """        
        try:
            wave, sr = sf.read(wave_path)
            wave_tensor = torch.from_numpy(wave).float()
        except Exception:
            print("ds")
        return wave_tensor

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self):
        """Constructor of the Collater
        """
        self.text_pad_index = 0
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            batch (List[__getitem__]): A list of sample obtained from __getitem__ function.

        Returns:
            (
                (N_Sample, NChannels, MelBand, T_Mel),
                (N_Sample,),
                (N_Sample, NChannels, MelBand, T_Mel),
                (N_Sample, NChannels, MelBand, T_Mel),
                (N_Sample,),
                (N_Sample, NChannels, MelBand, T_Mel),
                (N_Sample, NChannels, MelBand, T_Mel),
            ): Look at *__getitem__* DocString, in addition the __call__ add 2 element that are 2 random sampled vector representing emotion_encoding
        """ 
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
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels

def build_dataloader(dataset_path: str,
                     dataset_configuration: Dict,
                     batch_size:int = 4,
                     num_workers:int = 1,
                     device: str = 'cpu',
                     collate_config: dict = {}) -> DataLoader:
    """Make a dataloader

    Args:
        dataset_path (str): Path of the source dataset 
        dataset_configuration (Dict): Define if this dataloader will be used in a validation/test enviroment. Defaults to False.
        batch_size (int, optional): Batch Size. Defaults to 4.
        num_workers (int, optional): Number of Workers. Defaults to 1.
        device (str, optional): Device. Defaults to 'cpu'.
        collate_config (dict, optional): Flexible parameters. Defaults to {}.

    Raise
        FileNotFoundError: If the data_path is not a file
         
    Returns:
        DataLoader: The pytorch dataloader
    """  
        
    # Get Dataset info
    separetor = dataset_configuration["data_separetor"]
    data_header = dataset_configuration["data_header"]
    ####
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Check path! {dataset_path} does not exist!")

    dataset = pd.read_csv(dataset_path, sep=separetor, names=data_header)
    
    dataset = MelDataset(dataset)
    
    
    collate_fn = Collater(**collate_config)
    
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))
    

    return data_loader
