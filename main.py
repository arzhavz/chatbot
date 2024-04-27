import colorama
import os
import sys 
import time

from rich.console import Console
from rich.markdown import Markdown

from transformers import AutoTokenizer, AutoModelForCausalLM
from translate import Translator

def translate(text, lang):
    translate_client = Translator(to_lang=lang)
    translated_text = translate_client.translate(text)
    return translated_text

def display(string):
	for char in string:
		sys.stdout.write(char)
		sys.stdout.flush()
		time.sleep(0.05)
          
def generate_response(text, tokenizer, model, tl: bool = False, history: list = []):
    session = "".join(history)

    context = open("context.txt", "r").read()

    input_text = translate(text, "en") if tl else text
    template = f"<start_of_turn>user\n{context.strip()}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n"
    
    input_ids = tokenizer(session + template, return_tensors="pt").to("cpu")

    start_time = time.time()
    outputs = model.generate(**input_ids, max_new_tokens=256)
    end_time = time.time()

    raw_text = tokenizer.decode(outputs[0])
    out_text = raw_text.split(template)[1].split("<eos>")[0]
    execution_time = end_time - start_time
    
    return out_text, execution_time

def main():
    """
    Example inference for Google's Gemma AI.

    Gemma versions:
        - Gemma 2B:     google/gemma-2b-it
        - Gemma 1.1 2B: google/gemma-1.1-2b-it
    
    """

    session: list = []

    # model weight
    gemma: str = "google/gemma-1.1-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(gemma)
    model = AutoModelForCausalLM.from_pretrained(gemma, device_map="cpu")

    os.system("clear")

    console = Console()
    console.print(Markdown("# Firefly AI \n ## Gemma 1.1 2B base model, type 'exit' to quit chat session."))

    while True:
        input_text = input("{}You:{} ".format(colorama.Fore.CYAN, colorama.Style.RESET_ALL))

        if input_text == "exit":
            os.system("clear")
            break

        out_text, execution_time = generate_response(input_text, tokenizer, model, context = session)

        session.append(f"<start_of_turn>user\n{input_text}<end_of_turn>\n<start_of_turn>model\n{out_text.strip()}<end_of_turn>\n")

        display(out_text + "\n")
        print("\033[94m<Model execution time:", round(execution_time, 2), "seconds>\033[0m\n")

if __name__ == "__main__":
    main()
