import torch

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Using device CUDA\n')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print('Using device MPS\n')
else:
    print('Using device CPU\n')


MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

# One idea might be to force the model to perform latent thinking when the 
# perplexity on the next token falls below a certain threshold. 
# Ideally the model could also learn by itself when to perform latent thinking.
PERPLEXITY_THRESHOLD = 2.0 

# One naive way to make sure the model performs latent thinking is to force it to
# perform a certain number of steps before answering the query.
START_THINKING_STEPS = 30

# This will be changed depending on what we need to do
MAX_NEW_TOKENS = 1024



# Datasets
"""
GSM8K

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. 
The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.
"""




