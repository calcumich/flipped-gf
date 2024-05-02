from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm, trange
import torch

datapath = "dataset.json"
with open(datapath, 'r') as file:
    data = json.load(file)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading model")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.to(device)
resList = []
j = 0
ansListAnswerability = []
ansListCaptionIDs = []
print("starting evaluation loop")
for k in tqdm(data.keys()):
    maxL = 1200
    data_piece = data[k]
    question = data_piece['question']
    captions = data_piece['caption']
    captionList = [f for f in zip([k['id'] for k in captions], [k['text'] for k in captions])]
    #print([x for x in captionList])

    prompt = "### Task:\nEvaluate if the following question is answerable based on the video captions provided. If it is answerable, identify the captions that provide evidence for answering the question.\n\n"
    prompt += "### Question:\n" + question + "\n\n"
    prompt += "### Captions (ID, Caption)):\n"
    for index, caption in captionList:
        prompt += f"{index}: {caption}\n"
    prompt1 = prompt + "\n### Instructions:\nDetermine if the above captions provide enough information to answer the question.  Output 1 if the question can be answered, output 0 if the question cannot be answered."
    prompt2 = prompt + "\n### Instructions:\nDetermine the two captions that are most useful in answering the question.  Output the numbers of the two most useful captions:"
    input_ids = tokenizer(prompt1, return_tensors="pt").input_ids.to("cuda")
    #print(prompt2)
    maxL = len(prompt1) + 100
    outputs = model.generate(input_ids, max_length = maxL)
    ansAnswerability = tokenizer.decode(outputs[0])

    
    input_ids = tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda")
    maxL = len(prompt2) + 100
    outputs = model.generate(input_ids, max_length = maxL)
    ans = tokenizer.decode(outputs[0])
    #print(ans)
    
    resList.append({'id': k, 'answerability': ansAnswerability, 'evidence' : ans})
    if j + 1 % 3 == 0:
        print('\n\n\nhit the mystery\n\n\n')
        with open(f"output{j}AnswerabilityMistral.json", "w") as file:
            json.dump(ansListAnswerability, file)
        with open(f"output{j}IndicesMistral.json", "w") as file:
            json.dump(ansListAnswerability, file)
    j += 1
    if j > 5:
        break
with open(f"TestoutputFinalAnswerabilityMistral.json", "w") as file:
    json.dump(resList, file)
