from category_prompts import CATEGORIES

from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import hashlib
import numpy as np
import cv2
import torch

### IMAGE CONTENT HELPERS ###

device = "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_texts(categories):
    inputs = clip_processor(text=categories, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy()

# store (categories, embeddings)
TEXT_EMBED_CACHE = {
    key: (values, encode_texts(values))
    for key, values in CATEGORIES.items()
}


# image encoding
def clip_embedding(img_cv):
    try:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        inputs = clip_processor(images=img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**inputs)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        return img_features.squeeze().cpu().numpy()
    
    except Exception as e:
        print(f"error computing CLIP embedding: {e}")

        return np.zeros(512)

# help
_clip_classify_cache = {}

def _clip_classify(img_cv, categories, text_emb):
    try:
        img_hash = hashlib.md5(img_cv.tobytes()).hexdigest()
        if img_hash in _clip_classify_cache:
            img_emb = _clip_classify_cache[img_hash]
        else:
            img_emb = clip_embedding(img_cv)
            _clip_classify_cache[img_hash] = img_emb

        sims = cosine_similarity([img_emb], text_emb)[0]
        best_idx = int(np.argmax(sims))

        return categories[best_idx], float(sims[best_idx])
    
    except Exception as e:
        print(f"error in clip_classify: {e}")

        return None, 0.0


### CATEGORY HELPERS ###
def classify_subject_type(img_cv):
    cats, emb = TEXT_EMBED_CACHE["subject_type"]
    return _clip_classify(img_cv, cats, emb)

def classify_genre(img_cv):
    cats, emb = TEXT_EMBED_CACHE["genre"]
    return _clip_classify(img_cv, cats, emb)

def classify_perspective(img_cv):
    cats, emb = TEXT_EMBED_CACHE["perspective"]
    return _clip_classify(img_cv, cats, emb)

def classify_color_mode(img_cv):
    cats, emb = TEXT_EMBED_CACHE["color_mode"]
    return _clip_classify(img_cv, cats, emb)

def classify_complexity(img_cv):
    cats, emb = TEXT_EMBED_CACHE["complexity"]
    return _clip_classify(img_cv, cats, emb)
