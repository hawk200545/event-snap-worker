from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import base64
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

def get_embedding(image_path: str):
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Encode image as base64 data URL
    b64_image = base64.b64encode(image_data).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_image}"
    
    result = client.feature_extraction(
        data_url,
        model="sentence-transformers/clip-ViT-B-32"
    )
    
    return result

# Test it
embedding = get_embedding("C:\\Users\\abhis\\hackathon\\photos\\testimg.jpeg")
print("Success!")
print("Embedding length:", len(embedding))