from llm import load_model
from constants import DEVICE
from llm import LatentThinkingModel, BatchedLatentThinkingModel

if __name__ == "__main__":

    model = BatchedLatentThinkingModel()
    prompt1 = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."
    prompt2 = "Find the gradient of the following function $f(x,y) = (3x^2)*sin(x+y)$"
    prompt = [prompt1, prompt2]

    thought_result = model.generate_with_thinking(texts=prompt, pseudo_tokens=True, print_answer=True)
    standard_result = model.standard_model_generate(texts=prompt, print_answer=True)



    


