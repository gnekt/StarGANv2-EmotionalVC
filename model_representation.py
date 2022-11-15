#!/usr/bin/env python3
#coding:utf-8

from torch.utils.tensorboard import SummaryWriter
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, EmotionEncoder
from dataset_maker.emotion_mapping import emotion_map
import soundfile as sf
import random 
from meldataset import build_dataloader

EMOTION_LABEL=[id for id,value in emotion_map.items()]
MDOEL_PATH='Models/Experiment-1/epoch_00000.pth'
DEMO_PATH='Demo/neutral.mp3'
SAMPLE_RATE=24e3
SAMPLE_RATE=int(24e3)
DEVICE="cuda"
LOG_DIR="test"

config = yaml.safe_load(open("./Configs/config.yml"))

### Get configuration
batch_size = config.get('batch_size', 2)
device = config.get('device', 'cpu')
epochs = config.get('epochs', 1000)
save_freq = config.get('save_freq', 20)
dataset_configuration = config.get('dataset_configuration', None)
stage = config.get('stage', 'star')
fp16_run = config.get('fp16_run', False)
###
    
print("Start inference..")
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    emotion_encoder = EmotionEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     emotion_encoder=emotion_encoder)

    return nets_ema

def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = torch.LongTensor([speaker]).to(DEVICE)
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to(DEVICE), label)
        else:
            wave, sr = librosa.load(path, sr=SAMPLE_RATE)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave).to(DEVICE)

            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = starganv2.emotion_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    
    return reference_embeddings

writer = SummaryWriter(LOG_DIR + "/tensorboard/emotion_encoder_representation")

## load F0 model
print("Load f0 model..")
F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load("Utils/JDC/bst.t7")['net']
F0_model.load_state_dict(params)
_ = F0_model.eval()
F0_model = F0_model.to(DEVICE)

# load vocoder
print("Load vocoder model..")
from parallel_wavegan.utils import load_model
vocoder = load_model("Vocoder/PreTrainedVocoder/checkpoint-400000steps.pkl").to(DEVICE).eval()
vocoder.remove_weight_norm()
_ = vocoder.eval()

print("Load neural model..")
with open('Models/Experiment-1/config.yml') as f:
    starganv2_config = yaml.safe_load(f)
starganv2 = build_model(model_params=starganv2_config["model_params"])
params = torch.load(MDOEL_PATH, map_location='cpu')
params = params['model_ema']
_ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
_ = [starganv2[key].eval() for key in starganv2]
starganv2.emotion_encoder = starganv2.emotion_encoder.to(DEVICE)
starganv2.generator = starganv2.generator.to(DEVICE)

# load dataloader 
train_dataloader, val_dataloader = build_dataloader(dataset_configuration,
                                    batch_size=batch_size,
                                    num_workers=2,
                                    device=device)

for train_steps_per_epoch, batch in enumerate(train_dataloader):
    x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch
    # create grid of images
    # img_grid = writer.audio("audio", x_real)
    writer.add_graph(starganv2.emotion_encoder, [x_ref.to(DEVICE), y_trg.to(DEVICE)])
    writer.close()
    break