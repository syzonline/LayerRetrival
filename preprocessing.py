import os
import pandas as pd
import config

config = config.read_config()
caption_dir = config["captions_path"]

original_data = open('./dataset/new_caption.txt')
res = []
raw_id = 0
for item in original_data:
    raw = [x for x in item.strip().split('#')]
    line = []
    line.append(raw[0]) # image file
    line.append(raw_id) # caption number
    line.append(raw[2][2::]) # caption
    line.append(raw_id // 5) # id
    res.append(line)
    raw_id += 1
df = pd.DataFrame(columns=['image', 'caption_number', 'caption', 'id'], data=list(res))
path = os.path.join(caption_dir, "captions.csv")
df.to_csv(path)
