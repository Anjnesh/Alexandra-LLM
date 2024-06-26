import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer

model_name = "/content/mPLUG-Owl"  # path to the cloned repo
model = AutoModelForVision2Seq.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def describe_key_elements(image):
    inputs = tokenizer(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description
