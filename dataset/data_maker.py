import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import librosa
from emotion_mapping import emotion_map
import yaml

def sample(audio_path: str, sr=16000):
    """Down/Up Sample a specific audio file

    Args:
        audio_path (str): OS file path
        sr (int, optional): Sampling Rate expressed in Hz. Defaults to 16000.
    
    Raise
        FileNotFoundError: when audio_path is not found
    """    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Check path file, got {audio_path}")
    y, s = librosa.load(audio_path, sr=sr) # Downsample 44.1kHz to 8k
    
    
################################################
config_path="Configs/config.yml"
config = yaml.safe_load(open(config_path))

training_set_percentage = config.get('training_set_percentage', 80)
validation_set_percentage = config.get('validation_set_percentage', 20)

training_path = config.get('training_path', "Data/training_list.txt")
validation_path = config.get('validation_path', "Data/validation_list.txt" )
################################################

# Define stream 
dataframe = pd.read_csv("dataset/dataset.csv", sep=";").sample(frac=1)
training_file = open(training_path,"w")

# pick sample for training and validation
training_dataframe = dataframe.iloc[0:int((dataframe.shape[0]*training_set_percentage)/100)]
validation_dataframe = dataframe.iloc[dataframe.shape[0]-int((dataframe.shape[0]*validation_set_percentage)/100):]

# Create training file
for index, group in training_dataframe.groupby(["dataset","actor_id","statement_id"]):
    emotional_df = group[dataframe["emotion"] != "neutral"]
    try:
        neutral_row = group[dataframe["emotion"] == "neutral"].iloc[0]
    except IndexError:
        continue
    neutral_row['path'] = f"./dataset/{neutral_row['path'][2:]}"
    
    # neutral_row_path = f"./{neutral_row['lang']}/{neutral_row['dataset']}/{neutral_row['path'][2:]}"
    for index,row in emotional_df.iterrows():
        row['path'] = f"./dataset/{row['path'][2:]}"
        try:
            row['emotion']=emotion_map[row['emotion']]
        except KeyError as ex:
            continue
        try:
            emotional_row_path = f"./{row['lang']}/{row['dataset']}/{row['path'][2:]}"
            training_file.write(f"{neutral_row['actor_id']}|{neutral_row['statement_id']}|{neutral_row['path']}|0|{row['path']}|{row['emotion']}\n")
            training_file.write(f"{neutral_row['actor_id']}|{neutral_row['statement_id']}|{row['path']}|{row['emotion']}|{neutral_row['path']}|0\n")
        except IOError as e:
            print(e)
training_file.close()


validation_file = open(validation_path,"w")
# Create validation file
for index, group in validation_dataframe.groupby(["dataset","actor_id","statement_id"]):
    emotional_df = group[dataframe["emotion"] != "neutral"]
    try:
        neutral_row = group[dataframe["emotion"] == "neutral"].iloc[0]
    except IndexError:
        continue
    neutral_row['path'] = f"./dataset/{neutral_row['path'][2:]}"
    
    # neutral_row_path = f"./{neutral_row['lang']}/{neutral_row['dataset']}/{neutral_row['path'][2:]}"
    for index,row in emotional_df.iterrows():
        row['path'] = f"./dataset/{row['path'][2:]}"
        try:
            row['emotion']=emotion_map[row['emotion']]
        except KeyError as ex:
            continue
        try:
            emotional_row_path = f"./{row['lang']}/{row['dataset']}/{row['path'][2:]}"
            validation_file.write(f"{neutral_row['actor_id']}|{neutral_row['statement_id']}|{neutral_row['path']}|0|{row['path']}|{row['emotion']}\n")
            validation_file.write(f"{neutral_row['actor_id']}|{neutral_row['statement_id']}|{row['path']}|{row['emotion']}|{neutral_row['path']}|0\n")
        except IOError as e:
            print(e)
validation_file.close()

    