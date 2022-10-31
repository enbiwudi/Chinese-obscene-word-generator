import re
import torch
from unidecode import unidecode
from transformers import GPT2LMHeadModel


class MyTokenizer():
    def __init__(self):
        self.token2ids, self.ids2token = self.init_dict()
    
    def init_dict(self):
        token2ids = {}
        ids2token = {}
        with open('data/vocab.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip()
            token2ids[line] = len(token2ids)
            ids2token[len(token2ids) - 1] = line            
        return token2ids, ids2token
    
    def encode(self, text, max_len=64):
        input_ids = [101]
        output_ids = []
        for char in text:
            char_id = self.token2ids.get(char, self.token2ids.get(unidecode(char), self.token2ids.get(unidecode(char).lower(), 100)))
            input_ids.append(char_id)
            output_ids.append(char_id)
        output_ids.append(102)
        if max_len != None:
            input_ids.extend([0 for _ in range(max_len)])
            output_ids.extend([0 for _ in range(max_len)])
            input_ids, output_ids = input_ids[: max_len], output_ids[: max_len]
        return input_ids, output_ids
    
    def decode(self, ids):
        text = ''
        for id in ids:
            if id in [0]:
                continue
            text += self.ids2token[id]
        return text


def hell(text):
    softmax = torch.nn.Softmax(dim=-1)
    print('【' + text + '】：', end='')
    print(text, end='')
    while True:
        token = tokenizer.encode(text, max_len=None)[0]
        logits = model(torch.LongTensor(token).to('cuda')).logits
        top_prob, top_char = torch.sort(-softmax(logits), axis=-1)
        top_prob, top_char = -top_prob[:, :8].detach()[-1], top_char[:, :8].detach()[-1]
        try:
            next_char = np.random.choice(top_char.tolist(), p=top_prob.tolist()[:7] + [1 - sum(top_prob.tolist()[:7])])
        except:
            next_char = int(logits.argmax(-1)[-1])
        next_char = tokenizer.decode([next_char])
        text += next_char
        text = text[-63:]
        print(next_char, end='') if next_char != '[UNK]' else print('', end='')
        if next_char == '[SEP]' or len(text) > 512:
            break

if __name__ == '__main__':
    tokenizer = MyTokenizer()
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model.load_state_dict(torch.load('bad_boi.pt'))
    model.to('cuda')
    model.eval()
    hell(text = "我操你")
