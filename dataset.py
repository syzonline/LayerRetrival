import cv2
import albumentations as A
import torch
import config

config = config.read_config()
image_path = config["image_path"]
size = config["size"]

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and captions must have the same length,
        if there are multiple captions for each images,
        the image_filenames must have repetitive names.
        """
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=config["max_length"])
        self.transforms = transforms
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
        image = cv2.imread(f"{image_path}//{self.image_filenames[idx]}")
        # print(f"{image_path}/{self.image_filenames[idx]}")
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        
        return item
    
    def __len__(self):
        return len(self.captions)
    
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )