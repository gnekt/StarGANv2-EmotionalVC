# load packages
import time
from parallel_wavegan.utils import load_model
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
from models import Generator, StyleEncoder
from dataset.emotion_mapping import emotion_map
import soundfile as sf
import random
from typing import Dict, List

# DO NOT TOUCH
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4
######################################################################

# Variable
EMOTION_LABEL = [id for id, _ in emotion_map.items()]
MDOEL_PATH = 'Models/Experiment-2/ex_2_epoch.pth'
DEMO_PATH = 'Demo/neutral.wav'
SAMPLE_RATE = 24e3
SAMPLE_RATE = int(24e3)
DEVICE = "cuda"
######################################################################


def preprocess(wave_tensor: torch.Tensor) -> torch.Tensor:
    """Convert to Mel-Spectrogram

    Args:
        wave_tensor (sample,1): Waveform

    Returns:
        (MelBand, T_Mel): Mel-Spectrogram of the waveform
    """        
    wave_tensor = torch.from_numpy(wave_tensor).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def build_model(model_params={}) -> Munch:
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim,
                          w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    style_encoder = StyleEncoder(
        args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)

    nets_ema = Munch(generator=generator,
                     style_encoder=style_encoder)
    return nets_ema


def compute_style(speaker_dicts: Dict) -> Dict:
    """Compute emotion embedding for given audio reference

    Args:
        speaker_dicts (Dict): key: index, value: (Tuple) -> (audio path, emotion label)

    Returns:
        Dict(key: Tuple(torch.Tensor,torch.Tensor)): ...value: (Tuple) -> (embedding, emotion label)
    """    
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        wave, sr = librosa.load(path, sr=SAMPLE_RATE)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(DEVICE)
        with torch.no_grad():
            label = torch.LongTensor([speaker])
            ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)

    return reference_embeddings


# load F0 model
print("Load F0 model..")
F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load("Utils/JDC/bst.t7")['net']
F0_model.load_state_dict(params)
_ = F0_model.eval()
F0_model = F0_model.to(DEVICE)

# load vocoder
print("Load vocoder model..")
vocoder = load_model(
    "Vocoder/PreTrainedVocoder/checkpoint-400000steps.pkl").to(DEVICE).eval()
vocoder.remove_weight_norm()
_ = vocoder.eval()

print("Load neural model..")
with open('Models/Experiment-1/config.yml') as f:
    starganv2_config = yaml.safe_load(f)
starganv2 = build_model(model_params=starganv2_config["model_params"])
params = torch.load(MDOEL_PATH, map_location='cpu')
params = params['model_ema']
_ = [starganv2[key].load_state_dict(params[key]) for key in starganv2 if key!="mapping_network"]
_ = [starganv2[key].eval() for key in starganv2 if key!="mapping_network"]
starganv2.style_encoder = starganv2.style_encoder.to(DEVICE)
starganv2.generator = starganv2.generator.to(DEVICE)

# load input wave
audio, source_sr = librosa.load(DEMO_PATH, sr=SAMPLE_RATE)
audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32

# with reference, using style encoder
emotion_ref = {}
for index, val in emotion_map.items():
    if index == "neutral":
        continue
    emotion_ref[val] = (f'Demo/emotion_sample/{index}/{index}.wav', val)

print("computing reference embedding..")
reference_embeddings = compute_style(emotion_ref)


print("start generation..")
start = time.time()

source = preprocess(audio).to(DEVICE)
keys = []
converted_samples = {}
reconstructed_samples = {}
converted_mels = {}

for key, (ref, _) in reference_embeddings.items():
    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1)) # Out: (Batch, Channel, Features, T_Mel)
        # Is not sure that Real T_Mel is equal to T_Mel_Fake cause the fake comes from the generator
        out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat) # Out: (Batch, Channel, MelBand, T_Mel_Fake)

        c = out.transpose(-1, -2).squeeze().to(DEVICE)
        y_out = vocoder.inference(c)
        y_out = y_out.view(-1).cpu()

        if key not in emotion_ref or emotion_ref[key][0] == "":
            recon = None
        else:
            wave, sr = librosa.load(emotion_ref[key][0], sr=SAMPLE_RATE)
            mel = preprocess(wave)
            c = mel.transpose(-1, -2).squeeze().to(DEVICE)
            recon = vocoder.inference(c)
            recon = recon.view(-1).cpu().numpy()

    converted_samples[key] = y_out.numpy()
    reconstructed_samples[key] = recon

    converted_mels[key] = out

    keys.append(key)

end = time.time()
print('total processing time: %.3f sec' % (end - start))
for key, wave in converted_samples.items():
    emotion = EMOTION_LABEL[key]
    rnd_number = random.randint(1, 999)+random.randint(1, 999)
    print('Converted: %s' % key)
    print("storing sample..")
    sf.write(f'./Demo/out/{emotion}/{rnd_number}.wav', wave, SAMPLE_RATE)
