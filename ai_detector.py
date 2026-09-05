import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load GPT2 model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

model.eval()


def calculate_perplexity(text):

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])

    loss = outputs.loss
    perplexity = torch.exp(loss)

    return perplexity.item()


def ai_probability(text):

    perplexity = calculate_perplexity(text)

    # heuristic mapping
    if perplexity < 20:
        return 85
    elif perplexity < 35:
        return 65
    elif perplexity < 50:
        return 40
    else:
        return 15