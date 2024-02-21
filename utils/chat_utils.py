# Gradio Chat Interface for HuggingFace Hub Models 🚀 by Open Sistemas

"""
Created on Mon Dec  4 15:54:10 2023

@author: henry
"""


DEFAULT_SYSTEM_PROMPT = "You are a helpful bot created by the AI team of Open Sistemas, an innovative company in the are of AI & Data. Your answers are clear and concise and you must respond in the same language the user asked you."

LICENSE = """
<p/>

---
As a derivate work of Code Llama by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/codellama-2-13b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/codellama-2-13b-chat/blob/main/USE_POLICY.md).
"""

def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message



def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history


def delete_prev_fn(
        history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''


def support_system(tokenizer):
    try:
        # Attempt to use a system message with the tokenizer
        test_message = [{"role": "system", "content": "Test system support"}]
        tokenizer.apply_chat_template(test_message, tokenize=False)
        return True  # System messages are supported
    except:
        return False  # System messages are not supported

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
    
    # check if template supports system role
    if support_system(tokenizer):
        formatted_message = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]
    else:
        formatted_message = [
            {"role": "user", "content": "Who did create you?"},
            {"role": "assistant", "content": "I was created by the AI team of Open Sistemas, an innovative company in the are of AI & Data."}
        ]
    
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




