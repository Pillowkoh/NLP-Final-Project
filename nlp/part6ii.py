from multiprocessing.resource_sharer import stop
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
import copy
from collections import Counter, OrderedDict
from conlleval import evaluate, evaluate_conll_file
from part1 import *
from part2 import get_tags

def build_vocab(words):
    counter = Counter()
    for word_lst in words:
        word_lst = ['START'] + word_lst + ['STOP']
        counter.update(word_lst)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    return vocab(ordered_dict, specials=('START', 'STOP', 'PAD'))

def build_data(token_vocab, tag_vocab, train_dataset):
    data = []
    for sent in train_dataset:
        token_tensor = torch.LongTensor([token_vocab[token] for token, tag in sent])
        tag_tensor = torch.LongTensor([tag_vocab[tag] for token, tag in sent])
        
        # tag_tensor = F.one_hot(tag_tensor, num_classes=len(tag_vocab))
        data.append((token_tensor, tag_tensor))
    return data

def process_sentence(sent_tensor, start_idx, stop_idx, pad_idx):
    sent_tensor = torch.cat([torch.tensor([start_idx]), sent_tensor, torch.tensor([stop_idx])])
    if sent_tensor.shape[0] < SEQ_LENGTH:
        sent_tensor = torch.cat([sent_tensor, torch.tensor([pad_idx] * (SEQ_LENGTH - sent_tensor.shape[0]))])
    return sent_tensor

def process_text_data(train_data, start_idx, stop_idx, pad_idx, start_idx_tag, stop_idx_tag, pad_idx_tag):
    token_lens = [len(token_tensor) for token_tensor, tag_tensor in train_data]

    padded_token_tensors = [process_sentence(token_tensor, start_idx, stop_idx, pad_idx) for token_tensor, tag_tensor in train_data]
    padded_tag_tensors = [F.one_hot(process_sentence(tag_tensor, start_idx_tag, stop_idx_tag, pad_idx_tag), num_classes=len(train_tags_vocab)) for token_tensor, tag_tensor in train_data]
    return [(token_tensor, tag_tensor) for token_tensor, tag_tensor in zip(padded_token_tensors, padded_tag_tensors)]

'''
    LSTM Model architecture used for aspect-based sentiment analysis
'''
class ABSA_model(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_dim, hidden_dim, n_layers=3):
        super(ABSA_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_dim * 2, num_tags*SEQ_LENGTH)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x.view(x.shape[0], SEQ_LENGTH, self.num_tags, SEQ_LENGTH)
        x = self.sigmoid(x)
        return x[:, :, :, -1]

def train(model, optimizer, criterion, train_dataloader, epochs=250, early_stopping=5):
    early_stopping_losses = []
    models = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (token_batch, tag_batch) in enumerate(train_dataloader):
            token_batch, tag_batch = token_batch.to(device), tag_batch.to(device)
            model.zero_grad()
            output = model(token_batch)
            tag_batch = tag_batch.to(device, dtype=torch.float32)
            loss = criterion(output, tag_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        val_loss = 0.0
        model.eval()
        for i, (token_batch, tag_batch) in enumerate(val_dataloader):
            token_batch, tag_batch = token_batch.to(device), tag_batch.to(device)
            output = model(token_batch)
            tag_batch = tag_batch.to(device, dtype=torch.float32)
            loss = criterion(output, tag_batch)
            val_loss += loss.item()
        val_loss /= len(val_dataloader)
        print('Epoch: {}/{}'.format(epoch, epochs), ' Loss: {:.4f}'.format(train_loss), ' Val Loss: {:.4f}'.format(val_loss))
        
        # early stoppping
        models.append(copy.deepcopy(model))
        early_stopping_losses.append(val_loss)
        if len(early_stopping_losses) > 10:
            models.pop(0)
            early_stopping_losses.pop(0)
        if early_stopping_losses[-1] > early_stopping_losses[0]:
            print('Early stopping at epoch: {}'.format(epoch))
            model = models[np.argmax(early_stopping_losses)]
            break


def predict(model, tokens, train_vocab, train_tags_vocab, special_tags):
    start_idx, stop_idx, pad_idx = special_tags
    idx_to_str = train_tags_vocab.get_itos()

    padding_size = SEQ_LENGTH - len(tokens) - 2
    sent_tensor = torch.Tensor([train_vocab[token] for token in tokens])
    token_tensor = torch.unsqueeze(process_sentence(sent_tensor, start_idx, stop_idx, pad_idx), 0)
    token_tensor = token_tensor.to(device, dtype=torch.int32)

    model.eval()
    output = model(token_tensor)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)[1:-padding_size-1]

    output_idx = np.argmax(output, axis=1)
    tags = []
    for idx in output_idx:
        tags.append(idx_to_str[idx])
    return tags
    
def read_dev_in_file(filename):
    with open(filename, encoding='utf-8') as f:
        file_content = f.read()

    # Split the entire file into sentences. Output: List of sentences
    sentences = file_content.strip().split('\n\n')

    # Split each sentence into their tokens
    # Output: List of sentences. Each sentence is a list of tokens
    tokens = [i.split('\n') for i in sentences]

    return tokens

def predict_dev_in(model, filename, output_filename, train_vocab, train_tags_vocab, special_tags):
    sentences = []
    with open(filename) as file:
        lines = file.readlines()
        sentence = list()
        for line in lines:
            formatted_line = line.strip()   
            
            if(len(formatted_line) ==0):
                sentences.append(sentence)
                sentence = []
                continue
            sentence.append(formatted_line)

    with open(output_filename, "w") as wf:
        for sentence in sentences:
            pred = predict(model, sentence, train_vocab, train_tags_vocab, special_tags)
            for i in range(len(sentence)):
                wf.write(sentence[i] + " " + pred[i] + "\n")
                
            wf.write("\n")

if __name__ == '__main__':
    print('Reading train file...')
    train_dataset = read_train_file('./dataset/train')

    print('Processing data...')
    token_tags = [token_tag for sent in train_dataset for token_tag in sent]
    train_tokens = [[token for token, tag in sent] for sent in train_dataset]
    train_tags= [[tag for token, tag in sent] for sent in train_dataset]

    train_vocab, train_tags_vocab = build_vocab(train_tokens), build_vocab(train_tags)
    train_vocab.set_default_index(train_vocab['a'])
    train_tags_vocab.set_default_index(train_tags_vocab['PAD'])
    train_data = build_data(train_vocab, train_tags_vocab, train_dataset)

    TAGS = ['O', 'B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral']
    BATCH_SIZE = 32
    SEQ_LENGTH = 75
    START_TOKEN_IDX, STOP_TOKEN_IDX, PAD_TOKEN_IDX = train_vocab['START'], train_vocab['STOP'], train_vocab['PAD']
    START_TAG_IDX, STOP_TAG_IDX, PAD_TAG_IDX = train_tags_vocab['START'], train_tags_vocab['STOP'], train_tags_vocab['PAD']

    processed_data = process_text_data(train_data, start_idx=START_TOKEN_IDX, stop_idx=STOP_TOKEN_IDX, pad_idx=PAD_TOKEN_IDX,
                                       start_idx_tag=START_TAG_IDX, stop_idx_tag=STOP_TAG_IDX, pad_idx_tag=PAD_TAG_IDX)
    data_size = len(processed_data)
    processed_train_data = processed_data[:int(data_size * 0.8)]
    processed_val_data = processed_data[int(data_size * 0.8):]
    train_dataloader = DataLoader(processed_train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(processed_val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # model training
    print('Training model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ABSA_model(vocab_size=len(train_vocab), num_tags=len(train_tags_vocab), embedding_dim=100, hidden_dim=100, n_layers=2)
    model.to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train(model, optimizer, criterion, train_dataloader, epochs=250)
    
    # save model
    torch.save(model, 'model.pt')
    print('Saved model to model.pt')

    # model prediction  and evaluation
    print('Predicting on dataset/dev.in...')
    special_tags = (START_TOKEN_IDX, STOP_TOKEN_IDX, PAD_TOKEN_IDX)
    predict_dev_in(model, 'dataset/dev.in', 'dataset/dev.6ii.out', train_vocab, train_tags_vocab, special_tags)

    # model evaluation
    print('Running evaluation using conlleval...')
    g_tags, p_tags = get_tags('dataset/dev.p6ii.out', 'dataset/dev.out')
    print(evaluate(g_tags,p_tags,verbose=True))
    