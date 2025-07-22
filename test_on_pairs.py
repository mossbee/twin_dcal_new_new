import os
import json
import random
import torch
from src.utils.config import Config
from src.models.siamese_dcal import SiameseDCAL
from src.inference.predictor import DCALPredictor
from src.models.dcal_core import DCALEncoder

# === CONFIGURATION ===
CONFIG_PATH = "configs/kaggle_config.yaml"
TEST_PAIRS_PATH = "data/test_twin_pairs.json"
TEST_INFO_PATH = "data/test_dataset_infor.json"
CHECKPOINT_PATH = "/kaggle/input/nd-twin/checkpoint_epoch_0020.pth"  # <-- Set your checkpoint path here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD CONFIG ===
config = Config.load(CONFIG_PATH)

# === LOAD TEST PAIRS AND IMAGE PATHS ===
with open(TEST_PAIRS_PATH, "r") as f:
    test_pairs = json.load(f)
with open(TEST_INFO_PATH, "r") as f:
    test_info = json.load(f)

# === BUILD MODEL ===
# You may need to adjust these arguments based on your model definition and config
dcal_encoder = DCALEncoder(
    embed_dim=config.model.embed_dim,
    num_heads=config.model.num_heads,
    num_layers=config.model.num_layers,
    patch_size=config.data.patch_size,
    dropout=config.model.dropout,
    pretrained=config.model.pretrained,
    backbone=config.model.backbone,
)
model = SiameseDCAL(
    dcal_encoder=dcal_encoder,
    similarity_function="cosine",
    feature_dim=config.model.embed_dim,
    dropout=config.model.dropout,
    temperature=0.07,
    learnable_temperature=True,
)

# === LOAD CHECKPOINT ===
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if "model_state_dict" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# === CREATE PREDICTOR ===
predictor = DCALPredictor(model, config, device=DEVICE)

# === RUN INFERENCE ===
results = []
for id1, id2 in test_pairs:
    # Select one image for each ID (random or first)
    img1_path = random.choice(test_info[id1])
    img2_path = random.choice(test_info[id2])
    result = predictor.predict_pair(img1_path, img2_path)
    results.append({
        "id1": id1,
        "id2": id2,
        "img1": img1_path,
        "img2": img2_path,
        "similarity": result["similarity"],
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    })

# === PRINT OR SAVE RESULTS ===
for r in results:
    print(r)
# Optionally, save to a file:
# with open("test_results.json", "w") as f:
#     json.dump(results, f, indent=2) 