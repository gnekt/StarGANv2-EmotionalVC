import pandas as pd

dataframe = pd.read_csv("./dataset.csv", sep=";")
validation 
for index, group in dataframe.groupby(["dataset","actor_id","statement_id"]):
    emotional_df = group[dataframe["emotion"] != "neutral"]
    try:
        neutral_row = group[dataframe["emotion"] == "neutral"].iloc[0]
    except IndexError:
        continue
    # neutral_row_path = f"./{neutral_row['lang']}/{neutral_row['dataset']}/{neutral_row['path'][2:]}"
    for index,row in emotional_df.iterrows():
        try:
            # emotional_row_path = f"./{row['lang']}/{row['dataset']}/{row['path'][2:]}"
            out_file.write(f"{neutral_row['path']}|{row['path']}|{row['emotion']}\n")
        except IOError as e:
            print(e)
out_file.close()