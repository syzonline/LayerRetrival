import os

path = './dataset'
img_dir = './dataset/Flicker8k_images'
annotations = os.path.join(path, "flickr8kenc_caption.txt")
new_anns = os.path.join(path, "new_caption.txt")
dic = {}
global_id = 0
old_name = ''
with open(annotations, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name = line.strip().split('#')[0]
        if name != old_name:
            old_name = name
            new_name = str(global_id).zfill(4) + '.jpg'
            dic.update({name:new_name})
            global_id = global_id + 1

with open(annotations, 'r') as f:
    lines = f.readlines()
    for line in lines:
        key = line.strip().split('#')[0]
        if key in dic.keys():
            newline = line.replace(key, dic[key])
            with open(new_anns, 'a+') as fp:
                fp.write(newline)
            fp.close()
        else:
            print("Error!")
            
namelist = os.listdir(img_dir)
for item in namelist:
    os.rename(os.path.join(img_dir, item), os.path.join(img_dir, dic[item]))