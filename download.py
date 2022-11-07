# import gdown 
# pretrained_model = {
#     "PretrainedStarGanv2":["https://drive.google.com/uc?id=1ML42wB0kHmYe9Mpw5jEh6Grwdpx9QynV"],
#     "PreTrainedVocoder": ["https://drive.google.com/uc?id=1KoIOJiA4Zywm297uWdk1df9TDBxn-kX3"]
# }

# for index, value in pretrained_model.items():
#     gdown.download(value[0],f"{index}.zip")


import zipfile
with zipfile.ZipFile("datasets.zip", 'r') as zip_ref:
    zip_ref.extractall("./datasets")
