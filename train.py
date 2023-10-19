import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import torch
from transformers import DistilBertTokenizer
import config
import dataset
import util
import clip

config = config.read_config()
captions_path = config["captions_path"]
debug = config["debug"]
batch_size = config["batch_size"]
num_workers = config["num_workers"]
text_tokenizer = config["text_tokenizer"]
image_encoder_lr = config["image_encoder_lr"]
text_encoder_lr = config["text_encoder_lr"]
head_lr = config["head_lr"]
weight_decay = config["weight_decay"]
patience = config["patience"]
factor = config["factor"]
epochs = config["epochs"]

def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    print("train:", train_dataframe.shape, "val:", valid_dataframe.shape)
    return train_dataframe, valid_dataframe

def build_loaders(dataframe, tokenizer, mode):
    transforms = dataset.get_transforms(mode=mode)
    datasets = dataset.CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if mode == "train" else False,
    )
    print(len(dataframe["image"]), len(dataframe["image"][0]), len(dataframe["caption"][0]), dataframe["caption"].values)
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = util.AvgMeter()
    print("len:", len(train_loader))
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step == "batch":
            lr_scheduler.step()
            
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=util.get_lr(optimizer))
        
    return loss_meter
    
def valid_epoch(model, valid_loader):
    loss_meter = util.AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def test_epoch(model, test_df):
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    query = []
    for item in test_df["caption"]:
        id, caption = enumerate()
    encoded_query = tokenizer([query for query in test_df["caption"]])
    for item in test_df:
        encoded_query = tokenizer([])

def run_train():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = clip.CLIPModel().to("cuda" if torch.cuda.is_available() else "cpu")
    params = [
        {"params": model.image_encoder.parameters(), "lr": image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": head_lr, "weight_decay": weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "./save_space/best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

# Get accuracy after each train epoch using test set.
def test(recent_model):
    test_df = pd.read_csv(f"{captions_path}/test.csv")
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    test_loader = build_loaders(test_df, mode='test')
    model = clip.CLIPModel().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict()
        
if __name__ == "__main__":
    run_train()