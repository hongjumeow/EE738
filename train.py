#-*- coding: utf-8 -*-

import os
import json
import pdb
import argparse
import time
import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
import soundfile
import numpy as np
import editdistance
import pickle
from tqdm import tqdm
import re
from scipy import signal
import random
import glob
import librosa

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
            sample_rate=16000,
            n_mels=80,
            frame_length=20,
            frame_shift=10,
            normalize=True,
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
        self.normalize = normalize

    def __call__(self, signal):
        feature = self.transforms(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        ).numpy()
        if self.normalize:
            feature -= feature.mean()
            feature /= np.std(feature)
        
        input_length = feature.shape[0]
        return feature, input_length

def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-8)

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_path, rir_path, musan_path, max_length, char2index, trainmode):
        super(SpeechDataset, self).__init__()

        # load data from JSON
        with open(data_list,'r') as f:
            data = json.load(f)

        # convert seconds to frames
        max_length *= 16000

        self.augment = {
            'rir': False,
            'musan': False,
            'noise': False,
            'gain': False,
            'pitch': False,
            'specaug': True,
        }
        
        data = sorted(data, key=lambda d: d['len'], reverse=True)
        self.data = [x for x in data if x['len'] <= max_length]

        if self.augment['rir']:
            self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        
        if self.augment['musan']:
            self.noisetypes = ['noise','speech','music']

            self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
            self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
            self.noiselist  = {}
            augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
            for file in augment_files:
                if not file.split('/')[-3] in self.noiselist:
                    self.noiselist[file.split('/')[-3]] = []
                self.noiselist[file.split('/')[-3]].append(file)

        self.fbank = FBank()
        self.dataset_path   = data_path
        self.char2index     = char2index
        
        self.trainmode = trainmode

    def __getitem__(self, index):
        audio, sampling_rate = soundfile.read(os.path.join(self.dataset_path, self.data[index]['file']))

        if self.trainmode:
            if self.augment['rir']:
                if random.random() < 0.3:
                    audio = self.reverberate(audio)
            if self.augment['musan']:
                if random.random() < 0.5:
                    mode = random.choice(["music", "speech", "noise"])
                    audio = self.additive_noise(mode, audio)
            if self.augment['noise']:
                if random.random() < 0.2:
                    audio = torch.FloatTensor(audio)
                    noise_level = random.uniform(0.005, 0.02)
                    audio = audio + noise_level * torch.randn_like(audio)
            if self.augment['gain']:
                if random.random() < 0.2:
                    audio = torch.FloatTensor(audio)
                    gain = random.uniform(0.8, 1.2)
                    audio = audio * gain
            if self.augment['pitch']:
                if random.random() < 0.3:
                    audio = torch.FloatTensor(audio)
                    n_steps = random.uniform(-2.0, 2.0)
                    audio_np = audio.numpy()
                    shifted = librosa.effects.pitch_shift(audio_np, sr=sampling_rate, n_steps=n_steps)
                    audio = torch.from_numpy(shifted)

        feature, input_length = self.fbank(audio)
        
        feature = torch.Tensor(feature)
        if self.trainmode:
            if self.augment['specaug']:
                feature = self.spec_augment(feature)

        # read transcript and convert to indices
        transcript = self.data[index]['text']
        transcript = [self.special_filter(x) for x in transcript]
        transcript = self.parse_transcript(transcript)

        return feature, input_length, torch.LongTensor(transcript)

    def spec_augment(self, spec, time_mask_width=30, freq_mask_width=13, num_time_masks=1, num_freq_masks=1):
        spec = spec.clone()
        T, F = spec.shape[0], spec.shape[1]

        # Time masking
        for _ in range(num_time_masks):
            t = random.randint(0, time_mask_width)
            t0 = random.randint(0, max(1, T - t))
            spec[t0:t0 + t, :] = 0

        # Frequency masking
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_width)
            f0 = random.randint(0, max(1, F - f))
            spec[:, f0:f0 + f] = 0

        return spec

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        
        rir, _ = soundfile.read(rir_file)
        rir = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 
        clean_len = len(audio)

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio, _ = soundfile.read(noise)
            if len(noiseaudio) >= clean_len:
                noiseaudio = noiseaudio[:clean_len]
            else:
                repeat_times = int(np.ceil(clean_len / len(noiseaudio)))
                noiseaudio = np.tile(noiseaudio, repeat_times)[:clean_len]

            noise_snr = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4) 
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        
        total_noise = np.sum(noises, axis=0)
        mixed = audio + total_noise

        target_rms = rms(audio)
        mixed_rms = rms(mixed)
        if mixed_rms > 0:
            mixed = mixed * (target_rms / mixed_rms)

        return mixed

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
    (xx, input_lengths, yy) = zip(*batch)

    ## compute lengths of each item in xx and yy
    x_lens = [x.size(0) for x in xx]
    y_lens = [y.size(0) for y in yy]

    ## zero-pad to the longest length
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens, torch.IntTensor(input_lengths)

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

class MaskCNN(nn.Module):
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def forward(self, inputs, seq_lengths):
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def _get_sequence_lengths(self, module, seq_lengths):
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()

class Conv2dExtractor(nn.Module):
    def __init__(self, input_dim):
        super(Conv2dExtractor, self).__init__()
        self.input_dim = input_dim
        self.activation = nn.ReLU(inplace=True)
        self.conv = None

    def get_output_dim(self):
        factor = ((self.input_dim - 1) // 2 - 1) // 2
        output_dim = self.out_channels * factor

        return output_dim

    def forward(self, inputs, input_lengths):
        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)

        return outputs, output_lengths

class Conv2dSubsampling(Conv2dExtractor):
    def __init__(
            self,
            input_dim: int,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super(Conv2dSubsampling, self).__init__(input_dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
            )
        )

    def forward(self, inputs, input_lengths):
        outputs, input_lengths = super().forward(inputs, input_lengths)
        output_lengths = input_lengths >> 2
        output_lengths -= 1
        return outputs, output_lengths

class SpeechRecognitionModel(nn.Module):
    def __init__(self,
                 n_classes=11,
                 d_emb=256,
                 num_layers=4,
                 d_ffn=1024,
                 ):
        super(SpeechRecognitionModel, self).__init__()        
        self.d_emb = d_emb
        self.num_layers = num_layers
        self.d_ffn = d_ffn

        self.conv_subsample = Conv2dSubsampling(80, in_channels=1, out_channels=self.d_emb)
        self.input_projection = nn.Sequential(
            nn.Linear(self.conv_subsample.get_output_dim(), self.d_emb),
            nn.Dropout(p=0.2),
        )

        self.positional_encoding = RelPosEncXL(self.d_emb)
        self.conformer = ConformerEncoder(
            num_layers=self.num_layers,
            d_model=self.d_emb,
            d_ffn=self.d_ffn,
            nhead=4,
            kernel_size=15,
            activation=Swish,
            dropout=0.1,
            causal=False,
            attention_type="RelPosMHAXL",
        )

        self.classifier = nn.Linear(256,n_classes)

    def forward(self, x, input_lengths):
        x, _ = self.conv_subsample(x, input_lengths)
        x = self.input_projection(x)

        pos_embs = self.positional_encoding(x)
        x, _ = self.conformer(x, pos_embs=pos_embs)
        
        x = x.transpose(0, 1)
        x = self.classifier(x).log_softmax(dim=-1)
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
            x_lengths = data[4]

            output = model(x, x_lengths)
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
        audio, _ = soundfile.read(os.path.join(data_path, file['file']))
        feature, x_length = FBank()(audio)
        feature = torch.Tensor(feature).unsqueeze(0)
        x_lengths = torch.IntTensor([x_length])

        # forward pass through the model
        output = model(feature, x_lengths)

        # decode using the greedy decoder
        if use_greedy_decoder:
            pred = decoder(output)
            out_text = ''.join([index2char[x.item()] for x in pred])
        else:
            pred = decoder.decode(output.squeeze(1).detach().cpu().numpy())
            out_text = ''.join(pred)
        out_text = out_text.replace("_", "")

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
    parser.add_argument('--rir_path',   type=str, default='data/RIRS_NOISES/simulated_rirs')
    parser.add_argument('--musan_path', type=str, default='data/musan')

    ## related to training
    parser.add_argument('--max_epoch',  type=int, default=50,       help='number of epochs during training')
    parser.add_argument('--batch_size', type=int, default=20,      help='batch size')
    parser.add_argument('--lr',         type=int, default=5e-5,     help='learning rate')
    parser.add_argument('--seed',       type=int, default=2222,     help='random seed initialisation')
    
    parser.add_argument('--d_emb',      type=int, default=256,     help='dimension of embedding')
    parser.add_argument('--num_layers', type=int, default=4,       help='number of layers in the model')
    parser.add_argument('--d_ffn',      type=int, default=1024,     help='dimension of feedforward network')
    
    ## relating to loading and saving
    parser.add_argument('--initial_model',  type=str, default='',   help='load initial model, e.g. for finetuning')
    parser.add_argument('--save_path',      type=str, default='checkpoint',   help='location to save checkpoints')
    parser.add_argument('--use_greedy_decoder',   dest='use_greedy_decoder',    action='store_true', help='Decoder Type')
    parser.add_argument('--use_scheduler', dest='use_scheduler', action='store_true', help='Use learning rate scheduler')

    ## related to inference
    parser.add_argument('--eval',   dest='eval',    action='store_true', help='Evaluation mode')
    parser.add_argument('--gpu',    type=int,       default=0,      help='GPU index')

    args = parser.parse_args()

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)

    # load labels
    char2index, index2char = load_label_json(args.labels_path)

    ## make an instance of the model on GPU
    model = SpeechRecognitionModel(
        n_classes=len(char2index)+1,
        d_emb=args.d_emb,
        num_layers=args.num_layers,
        d_ffn=args.d_ffn,
        ).cuda()
    print('Model loaded. Number of parameters:',sum(p.numel() for p in model.parameters()))

    ## load from initial model
    if args.initial_model != '':
        model.load_state_dict(torch.load(args.initial_model))

    # make directory for saving models and output
    assert args.save_path != ''
    os.makedirs(args.save_path,exist_ok=True)

    ## code for inference - this uses val_path and val_list
    if args.eval:
        process_eval(model, args.val_path, args.val_list, index2char, save_path=args.save_path, use_greedy_decoder=args.use_greedy_decoder)
        quit()

    # initialise seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # define datasets
    trainset  = SpeechDataset(args.train_list, args.train_path, args.rir_path, args.musan_path, args.max_length, char2index, trainmode=True)
    valset    = SpeechDataset(args.val_list,   args.val_path,   args.rir_path, args.musan_path, args.max_length, char2index, trainmode=False)

    feature = trainset[0][0]

    import matplotlib.pyplot as plt
    plt.imshow(feature.numpy().T, aspect='auto', origin='lower')

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    ## set loss function with blank index
    criterion = nn.CTCLoss()

    ## initialise training log file
    f_log = open(os.path.join(args.save_path,'train.log'),'a+')
    f_log.write('{}\n'.format(args))
    f_log.flush()

    ## Train for args.max_epoch epochs
    for epoch in range(1, args.max_epoch+1):
        tloss = process_epoch(model, trainloader, criterion, optimizer, trainmode=True)
        vloss = process_epoch(model, valloader, criterion, optimizer, trainmode=False)

        if args.use_scheduler:
            scheduler.step()

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
