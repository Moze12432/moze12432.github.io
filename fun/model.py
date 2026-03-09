from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "moses132/dailydialog-chatbot"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_reply(message):

    inputs = tokenizer.encode(message, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return reply
