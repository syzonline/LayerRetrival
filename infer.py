import cv2
from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import config
import matplotlib.pyplot as plt
import clip
import train

config = config.read_config()
text_tokenizer = config["text_tokenizer"]
image_path = config["image_path"]

# Loading the model after training and returning the image embeddings with shape(val_size, 256) and the model itself.
def get_image_embeddings(valid_df, model_path='./save_space/best.pt'):
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    valid_loader = train.build_loaders(valid_df, tokenizer, mode='valid')
    
    # Load model.
    model = clip.CLIPModel().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval() # Set the model in evaluation mode.
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to("cuda" if torch.cuda.is_available() else "cpu"))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

# Finding matches.
# Model gets the inference model, image embeddings and a text query.
# Output: the most relevant images from the validation set.
def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    encoded_query = tokenizer([query])
    for key, values in encoded_query.items():
        print("key:", key, "value:", values)
    batch = {
        key: torch.tensor(values).to("cuda" if torch.cuda.is_available() else "cpu")
        for key, values in encoded_query.items()
    }
    for item in batch:
        print(item)
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()

if __name__ == "__main__":
    _, valid_df = train.make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df)
    find_matches(model, 
             image_embeddings,
             query="A man shows a fish to a child",
             image_filenames=valid_df['image'].values,
             n=9)