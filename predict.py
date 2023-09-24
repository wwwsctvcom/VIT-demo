import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor
from transformers import ViTForImageClassification, ViTConfig

if __name__ == "__main__":
    classes, id2label, label2id = ["cat", "dog"], {0: "cat", 1: "dog"}, {"cat": 0, "dog": 1}
    image_processor = AutoProcessor.from_pretrained(Path.cwd() / "model_name_or_path")
    model = ViTForImageClassification.from_pretrained(Path.cwd() / "model_trained" / "best",
                                                      id2label=id2label,
                                                      label2id=label2id)

    img = input('Input image filename:')
    image = Image.open(img)
    image_inputs = image_processor(image, return_tensors="pt").pixel_values
    label = int(torch.argmax(F.softmax(model(image_inputs).logits, dim=-1), dim=-1))
    print(label)
    print(id2label[label])
