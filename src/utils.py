import csv
from torch.utils.data import DataLoader,Dataset
import torch
from tqdm import tqdm


from config import Config
conf = Config()

label2id = {c:i for i,c in enumerate(conf.class_list)}

def load_raw_data(file_path):
    data = []
    with open(file_path,'r',encoding='utf-8')as f:
        for row in tqdm(csv.reader(f),desc=f'Loading{file_path}.'): #csv.reader(f)得到列表
            if len(row) < 2:
                continue
            text,label = row[0].strip(),row[1].strip()
            if not text or label not in label2id:
                continue
            data.append((text,label2id[label]))
    return data

#自定义 数据集合
class TextDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]

#回调函数
def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    encoded = conf.tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=conf.pad_size,
        return_tensors='pt'
    )
    return encoded['input_ids'],encoded['attention_mask'],labels

def build_dataloader():
    train_loader = DataLoader(TextDataset(load_raw_data(conf.train_path)),
                              batch_size=conf.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(TextDataset(load_raw_data(conf.test_path)),
                             batch_size=conf.batch_size, shuffle=False,
                             collate_fn=collate_fn)
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = build_dataloader()
    print(f'训练集批次数: {len(train_loader)}   样本数: {len(train_loader.dataset)}')
    input_ids, attention_mask, labels = next(iter(train_loader))
    print(input_ids.shape, attention_mask.shape, labels.shape)