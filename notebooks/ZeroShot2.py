from huggingface_hub import login
import json
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from tqdm import tqdm
import os
import random
import json
with open("config_zero_shot.json", "r") as f:
    config = json.load(f)

login(config['hf_token'])


device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def ask_question(question, context, tokenizer, model):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(answer)

tokenizer, model = setup_model()
model.to(device)



data_path = "data/star/STAR_val.json"
with open(data_path, "r") as file:
    my_data = json.load(file)
    
def _get_text(data):
    answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
    qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
    question = data["question"].capitalize().strip()
    if question[-1] != "?":
        question = str(question) + "?"

    options = {x['choice_id']: x['choice'] for x in data['choices']}
    options = [options[i] for i in range(len(options))]
    answer = options.index(data['answer'])

    q_text = f"Question: {question}\n"
    o_text = "Choices: \n"
    for i in range(len(options)):
        o_text += f"{answer_mapping[i]} {options[i]}\n"
    a_text = "Answer: The answer is "
    text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
    return text, answer
    
def ask_question(q_and_a, tokenizer, model):
    text = q_and_a[0]
    answer = q_and_a[1]
    i_text = "Instruction: Predict the answer based on the question.\n"
    q_text = text['q_text']
    o_text = text['o_text']
    a_text = text['a_text']
    s1 = i_text + q_text + o_text + a_text
    inputs = tokenizer(s1, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_length=150, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)
    return s1, answer

def go_getter(data, tokenizer, model):
    output_dir = "ZeroShotOutput"
    os.makedirs(output_dir, exist_ok=True)
    
    filtered_data = [d for d in data if not d['question_id'].startswith('Interaction')]
    random.shuffle(filtered_data)  

    res_dict = {}

    for inx, dat in enumerate(tqdm(filtered_data)):
        t_and_a = _get_text(dat)
        question_id = dat['question_id']
        video_id = dat['video_id']
        correct_answer = dat['answer']
        choices = dat['choices']
        print(f'question_id: {question_id}')
        print(f'video_id: {video_id}')
        print(f'correct_answer: {correct_answer}')
        prompt, ans = ask_question(t_and_a, tokenizer, model)
        res_dict[question_id] = {
            'video_id': video_id,
            'correct_answer': correct_answer,
            'choices': choices,
            'prompt': prompt,
            'gen_answer': ans
        }
        if inx % 50 == 0 and res_dict:
            output_path = os.path.join(output_dir, f"ZeroShotOutputIndex{inx}.json")
            with open(output_path, 'w') as f:
                json.dump(res_dict, f)
    if res_dict:
        output_path = os.path.join(output_dir, f"ZeroShotOutputFinal.json")
        with open(output_path, 'w') as f:
            json.dump(res_dict, f)
    return res_dict

test = go_getter(my_data, tokenizer, model)
with open("ZeroShotOutputFinal.json", "w") as f:
    json.dump(test, f)
