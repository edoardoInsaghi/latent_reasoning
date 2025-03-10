from datasets import load_dataset

dataset = load_dataset('gsm8k', 'main')

def extract_answer(example):

    answer_split = example['answer'].split('#### ')
    if len(answer_split) > 1:

        clean_answer = answer_split[-1].strip()
        return {
            'question': example['question'],
            'answer': clean_answer
        }
    return None 


GSM8K = dataset.map(
    extract_answer,
    remove_columns=['answer'],
    batched=False
).filter(lambda x: x is not None)




