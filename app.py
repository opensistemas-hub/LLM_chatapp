# Gradio Chat Interface for HuggingFace Hub Models 🚀 by Open Sistemas

"""
Run a Gradio chat interface with HuggingFace Hub model for conversational AI.

Usage:
    $ python chat_interface.py --model_name "meta-llama/Llama-2-7b-chat-hf"    # Model name
                                            "openchat/openchat_3.5"
                                            "stabilityai/stablelm-zephyr-3b"
                                            "Open-Orca/Mistral-7B-OpenOrca"
                                            "mistralai/Mistral-7B-Instruct-v0.1"
                                            "tiiuae/falcon-180B-chat"
                                            "Intel/neural-chat-7b-v3-1"
                                            "amazon/LightGPT"
                                            
Created on Wed Oct 25 16:20:27 2023
@author: henry
"""


import argparse
import torch
import gradio
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from utils.chat_utils import format_message

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generate a response from the Llama model
def get_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.

    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.

    Returns:
        str: Generated response from the Llama model.
    """
    
    # Setting message and history
    query = format_message(tokenizer, message, history)
    response = ""
    
    # Generation by model
    sequences = gen(query, 
                       max_new_tokens=opt.max_new_tokens, 
                       do_sample=True, 
                       temperature=opt.temperature, 
                       top_k=opt.top_k, 
                       top_p=opt.top_p)
    
    
    # This will empty the VRAM 
    torch.cuda.empty_cache()

    # Remove the prompt from the output 
    generated_text = sequences[0]['generated_text']
    response = generated_text[len(query):]  

    logging.info('Chatbot: ' + response.strip())
    return response.strip()


def parse_opt():
    parser = argparse.ArgumentParser(description='Run a Gradio chat interface with hugging face chat models.')
    parser.add_argument('--model-name', type=str, default='openchat/openchat_3.5', help='Model name of hugging face or local path')
    parser.add_argument('--memory-limit', type=int, default=4, help='Limit on how many past interactions to remember')
    parser.add_argument('--max-new-tokens', type=int, default=4096, help='Maximum new tokens to generate in the response')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--top-k', type=int, default=50, help='Top K sampling for generation')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top P sampling for generation')
    parser.add_argument('--server-port', type=int, default=5000, help='Port for running the Gradio server')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Server name for the Gradio interface')
    opt = parser.parse_args()
    logging.info(f'Arguments: {vars(opt)}')
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    
    # Set the global default compute type to float16
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model from huggingface hub or local path
    logging.info('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    #tokenizer.chat_template = # applyt chat template here available in folder chat templates
    model = AutoModelForCausalLM.from_pretrained(opt.model_name, device_map=0,  quantization_config=bnb_config)
    logging.info('Model succesfully loaded!')

    # define pipeline for text generation
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    gradio.ChatInterface(fn=get_response, 
                         chatbot=gradio.Chatbot(height=600),
                         ).launch(server_name='0.0.0.0', server_port = 5000, share = False)
