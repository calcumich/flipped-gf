import torch
from .base_dataset import BaseDataset
import json

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/star/STAR_{split}.json', 'r'))
        
        self.videofeatures = torch.load(f'./data/star/clipvitl14.pth')

        self.features = json.load(open(f'./data/star/features.json', 'r'))

        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
        self.num_options = 4
        print(f"Num {split} data: {len(self.data)}") 


    def _get_text(self, idx):
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
            
        options = {x['choice_id']: x['choice'] for x in self.data[idx]['choices']}
        options = [options[i] for i in range(self.num_options)]
        answer = options.index(self.data[idx]['answer'])
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer

    def _get_video(self, video_id, start, end):
        print(video_id)
        if video_id not in list(self.features.keys()):
            print(video_id)
            video = torch.zeros(1, self.features_dim)
            print("Star.py - Feature_dim",self.features_dim)
        else:
            oldvideo = self.videofeatures[video_id][start: end +1, :].float() # ts
            print("Start.py - Old video shape", oldvideo.shape)
            video = torch.tensor(self.features[video_id])          ############  Check cuda version, changed cuda from here
            print("Start.py - New video shape", video.shape)

        if len(video) > 10:
            sampled = []
            for j in range(10):
                sampled.append(video[(j * len(video)) // 10])
            video = torch.stack(sampled)
            video_len = 10
        elif len(video) < 10:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(10 - video_len, 512)], 0)
        else:
            video_len = 10
        
        if len(oldvideo) > 10:
            sampled = []
            for j in range(10):
                sampled.append(oldvideo[(j * len(oldvideo)) // self.max_feats])
            oldvideo = torch.stack(sampled)
            oldvideo_len = 10
        elif len(oldvideo) < 10:
            oldvideo_len = len(oldvideo)
            oldvideo = torch.cat([oldvideo, torch.zeros(10 - oldvideo_len, 768)], 0)
        else:
            oldvideo_len = 10
        
        video = torch.cat((oldvideo, video), dim=1)
        print("Start.py - New video shape", video.shape)
        #print(video_len)
        return video, video_len
    


    ######################################################### New function ##############################################################
    # def _get_video(self):
    #     # if video_id not in self.features:
    #     #     print(video_id)
    #     #     video = torch.zeros(1, self.features_dim)
    #     # else:
    #     #     video = torch.cat(self.features[video_id], dim = 0)
        
    #     video = torch.rand(5120,1)
    #     return video, 5120

    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        qtype = self.qtype_mapping[self.data[idx]['question_id'].split('_')[0]]
        text, answer = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        start, end = round(self.data[idx]['start']), round(self.data[idx]['end'])
        video, video_len = self._get_video(f'{vid}', start, end)
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}


    def __len__(self):
        return len(self.data)