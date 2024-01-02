# Gradio Chat Interface for HuggingFace Hub Models 🚀 by Open Sistemas

"""
Created on Mon Dec  4 15:54:10 2023

@author: henry
"""

import torch

# Formatting function for message and history
def format_message(tokenizer, message: str, history: list, memory_limit: int = 4) -> str:
    """
    Formats the message and history for the tokenizer model.

    Parameters:
        tokenizer: model tokenizer
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    
    formatted_message = [
        {"role": "system", "content": "You are a helpful bot created by the AI team of Open Sistemas, an innovative company in the are of AI & Data. Your answers are clear and concise and you must respond in the same language the user asked you."}
    ]
    
    # Mistral doesn't support system role
    # formatted_message = [
    #     {"role": "user", "content": "Who did create you?"},
    #     {"role": "assistant", "content": "I was created by the AI team of Open Sistemas, an innovative company in the are of AI & Data."}
    # ]
    
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        formatted_message.append({"role": "user", "content": message})
        return tokenizer.apply_chat_template(formatted_message, tokenize = False, add_generation_prompt=True)

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message.append({"role": "user", "content": user_msg})
        formatted_message.append({"role": "assistant", "content": model_answer})


    # Handle the current message
    formatted_message.append({"role": "user", "content": message})
    formatted_message = tokenizer.apply_chat_template(formatted_message, tokenize = False, add_generation_prompt=True)

    return formatted_message




