import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig
import config

config = config.read_config()
# model_name = config["model_name"]
pretrained = config["pretrained"]
trainable = config["trainable"]
text_encoder_model = config["text_encoder_model"]
projection_dim = config["projection_dim"]
dropout = config["dropout"]
temperature = config["temperature"]
image_embedding = config["image_embedding"]
text_embedding = config["text_embedding"]

# Image encoder, default setting is ResNet50.
# In ResNet50 case, the vector size will be 2048 after the nn.AdaptiveAvgPool2d() layer.
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
        self, model_name="./model/resnet50", pretrained=pretrained, trainable=trainable
    ):
        super().__init__()
        pretrained_cfg = timm.create_model('resnet50').default_cfg
        pretrained_cfg["file"] = "./model/resnet50/pytorch_model.bin"
        self.model = timm.create_model(
            "./model/resnet50", pretrained, num_classes=0, global_pool="avg", pretrained_cfg=pretrained_cfg
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

# We choose DistillBERT as default.
# In DistillBERT case, the output hidden representation for each token is a vector with size 768.
class TextEncoder(nn.Module):
    def __init__(self, model_name="./model/bert-eng", pretrained=pretrained, trainable=trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

# Encode both images(2048) and texts(768) into fixed size vectors.
# The dimension of sharing representation space is 256.
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projetction_dim=projection_dim, dropout=dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# Calculating constractive loss.
class CLIPModel(nn.Module):
    def __init__(self, temperature=temperature, image_embedding=image_embedding, text_embedding=text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.tempature = temperature
        
    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        logits = (text_embeddings @ image_embeddings.T) / self.tempature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.tempature, dim=-1)
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (texts_loss + images_loss) / 2.0
        return loss.mean()
    
# batch_size = 4
# dim = 256
# embeddings = torch.randn(batch_size, dim)
# out = embeddings @ embeddings.T
# print(F.softmax(out, dim=-1))
        