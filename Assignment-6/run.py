from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"

print("Loading model:", MODEL_ID)
processor = DonutProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(device)
model.eval()

#  Load image
IMAGE_PATH = r"C:\Users\draj0.CHANDRIKAV\Downloads\bill.jpeg"
image = Image.open(IMAGE_PATH).convert("RGB")

# Preprocess
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids.to(device)

with torch.no_grad():
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        num_beams=1,
        early_stopping=True,
    )

result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("\n=== RESULT ===")
print(result)
