#-*- coding: utf-8 -*-

import os
import json
import pdb
import argparse
import time
import torch
import torch.nn as nn
import torchaudio
import soundfile
import numpy as np
import editdistance
import pickle
from tqdm import tqdm
import librosa
import torchaudio
import re

from pyctcdecode import build_ctcdecoder
import kenlm
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.activations import Swish

## ===================================================================
## Load labels
## ===================================================================

def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char
            
        return char2index, index2char

## ===================================================================
## Data loader
## ===================================================================

class FBank(object):
    def __init__(
            self,
            sample_rate: int = 16000,
            n_mels: int = 80,
            frame_length: int = 20,
            frame_shift: int = 10
    ) -> None:
        try:
            import torchaudio
        except ImportError:
            raise ImportError("Please install torchaudio `pip install torchaudio`")
        self.transforms = torchaudio.compliance.kaldi.fbank
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, signal):
        return self.transforms(
            torch.Tensor(signal).unsqueeze(0),
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        ).numpy()

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_path, max_length, char2index):
        super(SpeechDataset, self).__init__()

        # load data from JSON
        with open(data_list,'r') as f:
            data = json.load(f)

        # convert seconds to frames
        max_length *= 16000

        # sort data in length order and filter data less than max_length
        data = sorted(data, key=lambda d: d['len'], reverse=True)
        self.data = [x for x in data if x['len'] <= max_length]
        self.fbank = FBank()

        self.dataset_path   = data_path
        self.char2index     = char2index

    def __getitem__(self, index):

        # read audio using soundfile.read
        audio, _ = librosa.load(os.path.join(self.dataset_path, self.data[index]['file']), sr=16000)
        feature = self.fbank(audio)
        
        feature -= feature.mean()
        feature /= np.std(feature)
        
        # read transcript and convert to indices
        transcript = self.data[index]['text']
        transcript = [self.special_filter(x) for x in transcript]
        transcript = self.parse_transcript(transcript)
        # print(transcript)

        return torch.Tensor(feature), torch.LongTensor(transcript)
        
    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        return transcript

    def special_filter(self, sentence):
        SENTENCE_MARK = ['?', '!', '.']
        NOISE = ['o', 'n', 'u', 'b', 'l']
        EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']
        new_sentence = str()
        for idx, ch in enumerate(sentence):
            if ch not in SENTENCE_MARK:
                if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                    continue

            if ch == '#':
                new_sentence += '샾'

            elif ch not in EXCEPT:
                new_sentence += ch

        pattern = re.compile(r'\s\s+')
        new_sentence = re.sub(pattern, ' ', new_sentence.strip())
        return new_sentence
    
    def __len__(self):
        return len(self.data)


## ===================================================================
## Define collate function
## ===================================================================

def pad_collate(batch):
    (xx, yy) = zip(*batch)

    ## compute lengths of each item in xx and yy
    x_lens = [x.size(0) for x in xx]
    y_lens = [y.size(0) for y in yy]

    ## zero-pad to the longest length
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens

## ===================================================================
## Define sampler 
## ===================================================================

class BucketingSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):

        # Shuffle bins in random order
        np.random.shuffle(self.bins)

        # For each bin
        for ids in self.bins:
            # Shuffle indices in random order
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

## ===================================================================
## Baseline speech recognition model
## ===================================================================
class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2   = nn.BatchNorm1d(out_channels)

        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_classes=11):
        super(SpeechRecognitionModel, self).__init__()
        self.use_resnet = False
        self.use_conformer = True


        if self.use_conformer:
            ## define the conformer layers
            self.input_proj = nn.Linear(80, 512)
            self.positional_encoding = RelPosEncXL(512)
            self.conformer = ConformerEncoder(
                num_layers=3,             # Match 3 LSTM layers
                d_model=512,              # Output dim (like 256x2 for BiLSTM)
                d_ffn=160,               # Feedforward dim (recommended: 4×d_model)
                nhead=4,                  # Attention heads (common: 8 for 512 dim)
                kernel_size=15,           # Local convolution kernel (ASR-friendly)
                activation=Swish,         # Swish is common for Conformers
                dropout=0.1,              # Same as your LSTM dropout
                causal=False,             # Set True for streaming ASR
                attention_type="RelPosMHAXL",  # Relative attention = better for ASR
            )
        else:
            if self.use_resnet:
                cnns = [
                    ResNetBlock1D(40, 64, kernel_size=3),     # input MFCC: (batch, 40, time)
                    ResNetBlock1D(64, 64),
                    ResNetBlock1D(64, 64),
                    ResNetBlock1D(64, 64),
                ]
            else:
                cnns = [
                        nn.Dropout(0.1),  
                        nn.Conv1d(40, 64, 3, stride=1, padding=1),
                        # nn.Conv1d(40, 64, 5, stride=2),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.1),  
                        nn.Conv1d(64,64,3, stride=1, padding=1),
                        # nn.Conv1d(64,64, 5, stride=2),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        # ResNetBlock1D(64, 64),
                ]
            
                for i in range(2):
                    cnns += [nn.Dropout(0.1),  
                            nn.Conv1d(64,64, 3, stride=1, padding=1),
                            nn.BatchNorm1d(64),
                            nn.ReLU()]

            ## define CNN layers
            self.cnns = nn.Sequential(*nn.ModuleList(cnns))
            ## define RNN layers as self.lstm - use a 3-layer bidirectional LSTM with 256 output size and 0.1 dropout
            self.lstm = nn.LSTM(64, 256, 3, bidirectional=True, dropout=0.1)

        ## define the fully connected layer
        self.classifier = nn.Linear(512,n_classes)

        self.preprocess   = torchaudio.transforms.MFCC(sample_rate=8000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

    def forward(self, x):

        # ## compute MFCC and perform mean variance normalisation
        # with torch.no_grad():
        #   x = self.preprocess(x)+1e-6
        #   x = self.instancenorm(x).detach()

        if self.use_conformer:
            # x = x.transpose(1, 2)
            x = self.input_proj(x)  # <--- Important! Project to 512 dim
            pos_embs = self.positional_encoding(x)  # <--- Important!
            x, _ = self.conformer(x, pos_embs=pos_embs)
            
            x = x.transpose(0, 1)  # Now: (T=299, B=20, C=1999)
        else:
            # pass the network through the CNN layers
            x = self.cnns(x)
            ## pass the network through the RNN layers - check the input dimensions of nn.LSTM()
            x = x.permute(2, 0, 1)
            x, _ = self.lstm(x)

        ## pass the network through the classifier
        x = self.classifier(x)

        return x


## ===================================================================
## Train an epoch on GPU
## ===================================================================

def process_epoch(model,loader,criterion,optimizer,trainmode=True):

    # Set the model to training or eval mode
    if trainmode:
        model.train()
    else:
        model.eval()

    ep_loss = 0
    ep_cnt  = 0

    with tqdm(loader, unit="batch") as tepoch:

        for data in tepoch:

            ## Load x and y
            x = data[0].cuda()
            y = data[1].cuda()
            y_len = torch.LongTensor(data[3])

            output = model(x)

            ## compute the loss using the CTC objective
            x_len = torch.LongTensor([output.size(0)]).repeat(output.size(1))
            loss = criterion(output, y, x_len, y_len)

            if trainmode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # keep running average of loss
            ep_loss += loss.item() * len(x)
            ep_cnt  += len(x)

            # print value to TQDM
            tepoch.set_postfix(loss=ep_loss/ep_cnt)

    return ep_loss/ep_cnt


## ===================================================================
## Greedy CTC Decoder
## ===================================================================

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """
        Given a sequence emission over labels, get the best path.
        """
        
        indices = torch.argmax(emission, dim=-1)  # [T]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if (i != self.blank)]

        return indices


## ===================================================================
## Evaluation script
## ===================================================================

def process_eval(model, data_path, data_list, index2char, save_path=None, use_greedy_decoder=True):

    # set model to evaluation mode
    model.eval()
    model.cpu()

    # initialise the greedy decoder
    if use_greedy_decoder:
        decoder = GreedyCTCDecoder(blank=len(index2char))
    else:
        decoder = build_ctcdecoder(
            labels=list(index2char.values()) + [''],  # add blank token
            kenlm_model_path='./kenlm-6.arpa'
        )

    # load data from JSON
    with open(data_list,'r') as f:
        data = json.load(f)

    results = []

    for file in tqdm(data):

        # read the wav file and convert to PyTorch format
        audio, sample_rate = soundfile.read(os.path.join(data_path, file['file']))
        feature = FBank()(audio)
        feature -= feature.mean()
        feature /= np.std(feature)
        feature = torch.Tensor(feature).unsqueeze(0)
        # audio = torch.FloatTensor(audio)

        # forward pass through the model
        output = model(feature) # (B, T, V)
        probs = torch.nn.functional.log_softmax(output, dim=-1)
        
        # decode using the greedy decoder
        if use_greedy_decoder:
            pred = decoder(output)
            out_text = ''.join([index2char[x.item()] for x in pred])
        else:
            # out_text = ''
            # for out in output:
            #     pred = decoder.decode(out.detach().cpu().numpy())
            #     for x in pred:
            #         assert x in index2char.values(), 'Prediction contains unknown character: {}'.format(x)
            #     out_text.join(pred)
            # # log_probs = torch.nn.functional.log_softmax(output, dim=-1).squeeze(1).detach().cpu().numpy()
            pred = decoder.decode(probs.squeeze(1).detach().cpu().numpy())
            for x in pred:
                assert x in index2char.values(), 'Prediction contains unknown character: {}'.format(x)
            # # pred = pred.replace("_", "")
            out_text = ''.join(pred)

            # out_text = ''
            # for p in probs:
            #     probs_np = p.detach().cpu().numpy()
            #     text = decoder.decode(probs_np)
            #     out_text.join(text)
        print(out_text)
        out_text = out_text.replace("_", "")
        # convert to text

        # print(out_text)
        
        # keep log of the results
        file['pred'] = out_text
        if 'text' in file:
            file['edit_dist']   = editdistance.eval(out_text.replace(' ',''),file['text'].replace(' ',''))
            file['gt_len']     = len(file['text'].replace(' ',''))
        results.append(file)
    
    # save results to json file
    with open(os.path.join(save_path,'results.json'), 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    # print CER if there is ground truth
    if 'text' in file:
        cer = sum([x['edit_dist'] for x in results]) / sum([x['gt_len'] for x in results])
        print('Character Error Rate is {:.2f}%'.format(cer*100))


## ===================================================================
## Main execution script
## ===================================================================

def main():

    parser = argparse.ArgumentParser(description='EE738 Exercise')

    ## related to data loading
    parser.add_argument('--max_length', type=int, default=10,   help='maximum length of audio file in seconds')
    parser.add_argument('--train_list', type=str, default='data/ks_train.json')
    parser.add_argument('--val_list',   type=str, default='data/ks_val.json')
    parser.add_argument('--labels_path',type=str, default='data/label.json')
    parser.add_argument('--train_path', type=str, default='data/kspon_train')
    parser.add_argument('--val_path',   type=str, default='data/kspon_eval')


    ## related to training
    parser.add_argument('--max_epoch',  type=int, default=10,       help='number of epochs during training')
    parser.add_argument('--batch_size', type=int, default=20,      help='batch size')
    parser.add_argument('--lr',         type=int, default=1e-4,     help='learning rate')
    parser.add_argument('--seed',       type=int, default=2222,     help='random seed initialisation')
    
    ## relating to loading and saving
    parser.add_argument('--initial_model',  type=str, default='',   help='load initial model, e.g. for finetuning')
    parser.add_argument('--save_path',      type=str, default='checkpoint',   help='location to save checkpoints')

    ## related to inference
    parser.add_argument('--eval',   dest='eval',    action='store_true', help='Evaluation mode')
    parser.add_argument('--gpu',    type=int,       default=0,      help='GPU index')

    args = parser.parse_args()

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)

    # load labels
    char2index, index2char = load_label_json(args.labels_path)

    ## make an instance of the model on GPU
    model = SpeechRecognitionModel(n_classes=len(char2index)+1).cuda()
    print('Model loaded. Number of parameters:',sum(p.numel() for p in model.parameters()))

    ## load from initial model
    if args.initial_model != '':
        model.load_state_dict(torch.load(args.initial_model))

    # make directory for saving models and output
    assert args.save_path != ''
    os.makedirs(args.save_path,exist_ok=True)

    ## code for inference - this uses val_path and val_list
    if args.eval:
        process_eval(model, args.val_path, args.val_list, index2char, save_path=args.save_path)
        quit()

    # initialise seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # define datasets
    trainset  = SpeechDataset(args.train_list, args.train_path, args.max_length, char2index)
    valset    = SpeechDataset(args.val_list,   args.val_path,   args.max_length, char2index)

    # initiate loader for each dataset with 'collate_fn' argument
    # do not use more than 6 workers
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_sampler=BucketingSampler(trainset, args.batch_size), 
        num_workers=4, 
        collate_fn=pad_collate,
        prefetch_factor=4)
    valloader   = torch.utils.data.DataLoader(valset,   
        batch_sampler=BucketingSampler(valset, args.batch_size), 
        num_workers=4, 
        collate_fn=pad_collate,
        prefetch_factor=4)

    ## define the optimizer with args.lr learning rate and appropriate weight decay
    # < fill your code here >
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    ## set loss function with blank index
    # < fill your code here >
    criterion = nn.CTCLoss()
    
    

    ## initialise training log file
    f_log = open(os.path.join(args.save_path,'train.log'),'a+')
    f_log.write('{}\n'.format(args))
    f_log.flush()

    ## Train for args.max_epoch epochs
    for epoch in range(0, args.max_epoch):

        # < fill your code here >
        tloss = process_epoch(model, trainloader, criterion, optimizer, trainmode=True)
        vloss = process_epoch(model, valloader, criterion, optimizer, trainmode=False)

        # save checkpoint to file
        save_file = '{}/model{:05d}.pt'.format(args.save_path,epoch)
        print('Saving model {}'.format(save_file))
        torch.save(model.state_dict(), save_file)

        # write training progress to log
        f_log.write('Epoch {:03d}, train loss {:.3f}, val loss {:.3f}\n'.format(epoch, tloss, vloss))
        f_log.flush()

    f_log.close()


if __name__ == "__main__":
    main()
