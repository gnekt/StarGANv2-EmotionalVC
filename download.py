import gdown 
pretrained_model = {
    "dataset":["https://drive.google.com/uc?id=1hQqfuvLe03G6Xx4B-t1uuRBbbNG7epCh"],
    "PreTrainedVocoder": ["https://drive.google.com/uc?id=1KoIOJiA4Zywm297uWdk1df9TDBxn-kX3"]
}
for index, value in pretrained_model.items():
    gdown.download(value[0],f"{index}.zip")

import zipfile
with zipfile.ZipFile("./dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("./dataset")
