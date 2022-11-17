### script that try to evaluate emotion embedding
# load packages
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
import pandas as pd

EMOTION_LABEL=[id for id,value in emotion_map.items()]
MDOEL_PATH='Models/Experiment-2/epoch_00040.pth'
DEMO_PATH='Demo/neutral.mp3'
SAMPLE_RATE=24e3
SAMPLE_RATE=int(24e3)
DEVICE="cuda"

emotion_map = {
    "neutral":0,
    "anger":1,
    "happy":2,
    "sad":3
}


print("Start tester..")

complete_dataset = pd.read_csv("Data/dataset.txt", sep='|', names=["actor_id","statement_id","source_path","source_emotion","reference_path","reference_emotion"])
anger_batch = complete_dataset[complete_dataset["reference_emotion"]=="1"].sample(n=20)
happy_batch = complete_dataset[complete_dataset["reference_emotion"]=="2"].sample(n=20)
sad_batch = complete_dataset[complete_dataset["reference_emotion"]=="3"].sample(n=20)
dataset = pd.concat([anger_batch,happy_batch,sad_batch])

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
    for key, (path, emotion) in speaker_dicts.items():
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

# load F0 model
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
with open('Models/Experiment-2/config.yml') as f:
    starganv2_config = yaml.safe_load(f)
starganv2 = build_model(model_params=starganv2_config["model_params"])
params = torch.load(MDOEL_PATH, map_location='cpu')
params = params['model_ema']
_ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
_ = [starganv2[key].eval() for key in starganv2]
starganv2.emotion_encoder = starganv2.emotion_encoder.to(DEVICE)
starganv2.generator = starganv2.generator.to(DEVICE)

# style encoder
encoding = []
for index,row in dataset.iterrows():
    wave, sr = librosa.load(row["reference_path"], sr=SAMPLE_RATE)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        wave = librosa.resample(wave, sr, 24000)
    mel_tensor = preprocess(wave).to(DEVICE)
    with torch.no_grad():
        label = torch.LongTensor([row[""]])
        ref = starganv2.emotion_encoder(mel_tensor.unsqueeze(1), label)
