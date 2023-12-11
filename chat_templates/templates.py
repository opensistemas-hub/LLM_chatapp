# Gradio Chat Interface for HuggingFace Hub Models 🚀 by Open Sistemas

"""
Created on Mon Dec  4 15:54:10 2023

@author: henry
"""

# Any Amazon lightGPT model: https://huggingface.co/amazon/LightGPT
amazon_light_gpt = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n' }}{% elif message['role'] == 'user' %}{{ '\n### Instruction:\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '\n### Response:\n'  + message['content'] + ''}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n### Response:\n' }}{% endif %}"

# Any neural chat family model: https://huggingface.co/Intel/neural-chat-7b-v3-1
intel_neural_chat = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '### System:\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '### User:\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n'  + message['content'] + ''}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}"

# Any neural chat family model: https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
together_redpajama = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<human>: ' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<bot>: '  + message['content'] + ''}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<bot>: ' }}{% endif %}"