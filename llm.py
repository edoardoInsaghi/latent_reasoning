import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MODEL_NAME, DEVICE, START_THINKING_STEPS

def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer


class LatentThinkingModel(nn.Module):

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.model, self.tokenizer = load_model(model_name)
        self.thinking_state = False

    
    def _format_input(self, text_prompt):

        messages = [
            {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
            {"role": "user", "content": text_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_length = len(text)
        token_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        latent_inputs = self.model.get_input_embeddings()(token_inputs.input_ids)
        attention_mask = token_inputs.attention_mask

        return token_inputs, latent_inputs, attention_mask, prompt_length


    def generate_with_thinking(self, text, max_new_tokens=256, pseudo_tokens=True):

        token_inputs, input_embeds, attention_mask, prompt_length = self._format_input(text)

        self.thinking_state = True
        latent_thoughts = []
        pseudo_tokens_list = []
        generated = 0

        # Thinking loop before generating output
        while self.thinking_state:

            out = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            latent_outputs = out.hidden_states[-1]
            latent_thoughts.append(latent_outputs)

            if pseudo_tokens:
                logits = self.model.lm_head(latent_outputs[:, -1, :])
                pseudo_token_id = torch.argmax(logits, dim=-1).item()
                pseudo_tokens_list.append(pseudo_token_id)
            
            # Append new latent token and update mask
            input_embeds = torch.cat([input_embeds, latent_outputs[:, -1:, :]], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=DEVICE)], dim=1)

            # TODO: check if transformers allow to return logits for each token in vocabsize
            # Now we just get the latent of the token that would be generated, ideally
            # we could mix the latent of the top k tokens or something else idk

            generated += 1
            if generated >= START_THINKING_STEPS:
                self.thinking_state = False

        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )

        if pseudo_tokens:
            full_sequence = (
                token_inputs.input_ids[0].tolist() +  
                pseudo_tokens_list +           
                outputs[0].tolist()
            )
            full_sequence = self.tokenizer.decode(full_sequence, skip_special_tokens=False)

        else: 
            full_sequence = torch.cat([token_inputs.input_ids[0], outputs[0]], dim=-1)
            full_sequence = self.tokenizer.decode(full_sequence, skip_special_tokens=False)
            full_sequence = full_sequence[:prompt_length] + "\n\n" + "|thinking|" * START_THINKING_STEPS + "\n\n" + full_sequence[prompt_length:]

        return full_sequence
    

    def standard_model_generate(self, text, max_new_tokens):
        token_inputs, _, _, _ = self._format_input(text)
        outputs = self.model.generate(
            **token_inputs,
            max_new_tokens=max_new_tokens
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            

            




