# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
import torchaudio
import librosa
from models import Generator, EmotionEncoder
from dataset_maker.emotion_mapping import emotion_map
import soundfile as sf
import random 
from typing import List, Dict, Tuple

# DO NOT TOUCH
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4
###########################################################

# Variable
EMOTION_LABEL=[id for id, _ in emotion_map.items()]
MDOEL_PATH='Models/Experiment-3/ex_3_epoch.pth'
DEMO_PATH='Demo/neutral.wav'
SAMPLE_RATE=24e3
SAMPLE_RATE=int(24e3)
DEVICE="cuda"
##########################################################

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
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf)
    emotion_encoder = EmotionEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     emotion_encoder=emotion_encoder)

    return nets_ema

def compute_style(speaker_dicts: Dict) -> torch.Tensor:
    """Compute emotion embedding for given audio reference

    Args:
        speaker_dicts (Dict): key: index, value: (Tuple) -> (audio path, emotion label)

    Returns:
        (Batch, Embedding Dim): The embeddings
    """    
    inputs = torch.zeros((len(speaker_dicts.items()),1,80,192))
    label = torch.zeros(len(speaker_dicts.items())).type(torch.LongTensor)
    for counter,(key, (path, speaker)) in enumerate(speaker_dicts.items()):
        wave, sr = librosa.load(path, sr=24000)
        wave, index = librosa.effects.trim(wave, top_db=30)
        
        if sr != 24000:
            wave = librosa.resample(wave, sr, 24000)
        mel = preprocess(wave).to(DEVICE)
        # If the audio reference is longer than T_Mel*Hop_Len(300) / 24e3 -> 2.4second
        if mel.shape[2] >= 192: mel = mel[:,:,:192] 
        inputs[counter,0,:,:mel.shape[2]] = mel
        label[counter] = speaker
    with torch.no_grad():
        label = label.to("cuda")
        embeddings = starganv2.emotion_encoder(inputs.to("cuda"), label)
    
    return embeddings

# load vocoder
print("Load vocoder model..")
from parallel_wavegan.utils import load_model
vocoder = load_model("Vocoder/PreTrainedVocoder/checkpoint-400000steps.pkl").to(DEVICE).eval()
vocoder.remove_weight_norm()
_ = vocoder.eval()

# load neural model
print("Load neural model..")
with open('Models/Experiment-3/config.yml') as f:
    starganv2_config = yaml.safe_load(f)
starganv2 = build_model(model_params=starganv2_config["model_params"])
params = torch.load(MDOEL_PATH, map_location='cuda')
params = params['model_ema']
_ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
_ = [starganv2[key].eval() for key in starganv2]
starganv2.emotion_encoder = starganv2.emotion_encoder.to(DEVICE)
starganv2.generator = starganv2.generator.to(DEVICE)

# load input wave
audio, source_sr = librosa.load(DEMO_PATH, sr=SAMPLE_RATE)
audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32

# with reference, using style encoder
emotion_ref={}
for index,val in emotion_map.items():
    print(index)
    if index=="neutral": continue
    emotion_ref[val] = (f'Demo/emotion_sample/{index}/{index}.wav', val)

reference_embeddings = compute_style(emotion_ref)


# conversion 
import time
start = time.time()
    
source = preprocess(audio).to(DEVICE)
keys = []
converted_samples = {}
reconstructed_samples = {}
converted_mels = {}


converted_samples = {
    "anger":1,
    "happy":2,
    "sad":3
}

with torch.no_grad():
        out = starganv2.generator(source.unsqueeze(1), reference_embeddings)
        c = out.transpose(-1, -2).squeeze().to(DEVICE)
        for i,(emotion,value) in enumerate(converted_samples.items()):
            y_out = vocoder.inference(c[i])
            y_out = y_out.view(-1).cpu()
            converted_samples[emotion]=y_out
            
end = time.time()
print('total processing time: %.3f sec' % (end - start) )

for key, wave in converted_samples.items():
    emotion=key
    rnd_number=random.randint(1,999)+random.randint(1,999)
    print('Converted: %s' % key)
    print("storing sample..")
    sf.write(f'./Demo/out/{emotion}/{rnd_number}.wav', wave, SAMPLE_RATE)
