import time
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration

# prepare VILT model
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare BLIP model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# image
image = Image.open("./images/food.jpeg").convert("RGB")

# first question
text = input("Human> ")

# function to answer questions about an image
def get_vqa_answer(text):
    # prepare inputs
    encoding = vilt_processor(image, text, return_tensors="pt")

    # forward pass
    outputs = vilt_model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return vilt_model.config.id2label[idx];

# function to provide a caption for an image
def get_img_caption():
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs, max_new_tokens=1000)
    return blip_processor.decode(out[0], skip_special_tokens=True)

while text != 'q':
    # print the answer
    start = time.time()
    caption = get_img_caption();
    answer = get_vqa_answer(text)
    end = time.time()

    print(f"Bot [{round(end-start, 2)} s]>", answer)

    # Take the next question
    text = input("Human> ")