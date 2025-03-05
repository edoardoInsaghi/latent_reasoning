from llm import load_model
from constants import DEVICE
from llm import LatentThinkingModel

if __name__ == "__main__":

    model = LatentThinkingModel()
    prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

    thought_result = model.generate_with_thinking(text=prompt, max_new_tokens=512, pseudo_tokens=False)
    print(f"\n\nResult from the latent thinking model:\n {thought_result}\n\n")

    standard_result = model.standard_model_generate(text=prompt, max_new_tokens=512)
    print(f"\n\nResult from the standard model:\n {standard_result}\n\n")

    


