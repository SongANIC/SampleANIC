'''
Created on 2021-03-14 10:45:00

@Author: Kai Wang，xinghui Song
'''
import os
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5"
# Data parameters
data_folder = '../../data/OutputDataset/'  # folder with data files saved by create_input_files.py
data_name1 = 'flickr5_min_word_freq'  # base name shared by data files
data_name2 = 'PCCD5_min_word_freq'
# data_folder = '/data/fashion/'  # folder with data files saved by create_input_files.py
# data_name = 'fashion_3_cap_per_img_5_min_word_freq'  # base name shared by data files
print('data_folder: {}'.format(data_folder))
# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
# cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法

# Training parameters
start_epoch = 13
epochs = 40 # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 16
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-5  # learning rate for encoder if fine-tuning
decoder_lr = 4e-5  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = './models'  # path to checkpoint, None if none

def train(train_loader1, train_loader2, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, save_epochs, models_path):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    balance = False      # True（少的复制自身），False（多的去掉多余的）
    
    # 记录 LOSS 和 ACCS
    LOSS = []
    ACCS = []

    start = time.time()

    if balance:
        trains1 = [] # 原数据的数据量比较多
        trains2 = [] # 原数据的数据量比较少
        # 遍历两个loader
        for (imgs1, caps1, caplens1) in train_loader1: 
            # 分别将数据装入列表
            trains1.append((imgs1, caps1, caplens1))
        for (imgs2, caps2, caplens2) in train_loader2: 
            # 分别将数据装入列表
            trains2.append((imgs2, caps2, caplens2)) 
        
        # 对较少的那份复制自身进行补充
        for each in trains2[:(len(train_loader1) - len(train_loader2))]:
            trains2.append(each)

        for i, ((imgs1, caps1, caplens1), (imgs2, caps2, caplens2)) in enumerate(zip(trains1, trains2)):
            if imgs1.shape[0] != imgs2.shape[0]:
                break
            data_time.update(time.time() - start)
            # Move to GPU, if available
            imgs1 = imgs1.to(device)
            caps1 = caps1.to(device)
            caplens1 = caplens1.to(device)

            # Move to GPU, if available
            imgs2 = imgs2.to(device)
            caps2 = caps2.to(device)
            caplens2 = caplens2.to(device)
            
            # Forward prop.
            h1 = encoder(imgs1)
            h2 = encoder(imgs2)
            h  = torch.cat((h1, h2), axis=3)
            
            # "<start>": 8549, "<end>": 8550, "<pad>": 0, ".": 16
            # 将caps1的end置未pad， 将caps2的start置为pad，然后将caps1和caps2的有效部分concat起来，这样便只有一个start和一个end
            caps = torch.zeros(size=(caps1.shape[0], caps1.shape[1]*2), dtype=caps1.dtype).cuda()
            for idx, (cap1, cap2) in enumerate(zip(caps1, caps2)):
                cap1_end_idx = cap1.tolist().index(8550)
                cap2_end_idx = cap2.tolist().index(8550)
                
                caps[idx][:cap1_end_idx] = cap1[:cap1_end_idx]
                caps[idx][cap1_end_idx:(cap1_end_idx) + (cap2_end_idx)] = cap2[1:cap2_end_idx+1]

            caplens = caplens1 + caplens2 - 2

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(h, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Calculate loss
            #print('scores:', scores.shape, scores)
            #print('targets:', targets.shape, targets)
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i+1, min(len(trains1),len(trains2)),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            mylog = open('record.log',mode='a',encoding='utf-8')
            print('Epoch:',epoch,file=mylog)                                                              
            LOSS.append(loss.item())
            ACCS.append(top5)
            
        
    else:
        for i, ((imgs1, caps1, caplens1), (imgs2, caps2, caplens2)) in enumerate(zip(train_loader1, train_loader2)):
            if imgs1.shape[0] != imgs2.shape[0]:
                break
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs1 = imgs1.to(device)
            caps1 = caps1.to(device)
            caplens1 = caplens1.to(device)

            # Move to GPU, if available
            imgs2 = imgs2.to(device)
            caps2 = caps2.to(device)
            caplens2 = caplens2.to(device)
            
            # Forward prop.
            h1 = encoder(imgs1)
            h2 = encoder(imgs2)
            h  = torch.cat((h1, h2), axis=3)

            # "<start>": 8549, "<end>": 8550, "<pad>": 0, ".": 16
            # 将caps1的end置未pad， 将caps2的start置为pad，然后将caps1和caps2的有效部分concat起来，这样便只有一个start和一个end
            caps = torch.zeros(size=(caps1.shape[0], caps1.shape[1]*2), dtype=caps1.dtype).cuda()
            for idx, (cap1, cap2) in enumerate(zip(caps1, caps2)):
                cap1_end_idx = cap1.tolist().index(8550)
                cap2_end_idx = cap2.tolist().index(8550)
                
                caps[idx][:cap1_end_idx] = cap1[:cap1_end_idx]
                caps[idx][cap1_end_idx:(cap1_end_idx) + (cap2_end_idx)] = cap2[1:cap2_end_idx+1]
                
            caplens = caplens1 + caplens2 - 2

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(h, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Calculate loss
            #print('scores:', scores.shape, scores)
            #print('targets:', targets.shape, targets)
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i+1, min(len(train_loader1),len(train_loader2)),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            LOSS.append(loss.item())
            ACCS.append(top5)
    
    # 在指定的epoch保存模型
    if epoch in save_epochs:
        encoder_name = 'encoder_saved_epoch_{}'.format(epoch)
        encoder_saved_path = os.path.join(models_path, encoder_name)
        torch.save(encoder, encoder_saved_path)

        decoder_name = 'decoder_saved_epoch_{}'.format(epoch)
        decoder_saved_path = os.path.join(models_path, decoder_name)
        torch.save(decoder, decoder_saved_path)
            
    return LOSS, ACCS, loss.item(), top5
    

def validate(val_loader1, val_loader2, encoder, decoder, criterion):

    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder modelval_loader2,
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    # 记录 LOSS 和 ACCS
    LOSS = []
    ACCS = []

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    balance = True
    with torch.no_grad():
        if balance:
            vals1 = [] # 原数据的数据量比较多
            vals2 = [] # 原数据的数据量比较少
            # 遍历两个loader
            for (imgs1, caps1, caplens1, _) in val_loader1: 
                # 分别将数据装入列表
                vals1.append((imgs1, caps1, caplens1))
            for (imgs2, caps2, caplens2, _) in val_loader2: 
                # 分别将数据装入列表
                vals2.append((imgs2, caps2, caplens2)) 
            
            # 对较少的那份复制自身进行补充
            for each in vals2[:(len(val_loader1) - len(val_loader2))]:
                vals2.append(each)
            
            for i, ((imgs1, caps1, caplens1), (imgs2, caps2, caplens2)) in enumerate(zip(vals1, vals2)):
                if imgs1.shape[0] != imgs2.shape[0]:
                    break
                # Move to GPU, if available
                imgs1 = imgs1.to(device)
                caps1 = caps1.to(device)
                caplens1 = caplens1.to(device)

                # Move to GPU, if available
                imgs2 = imgs2.to(device)
                caps2 = caps2.to(device)
                caplens2 = caplens2.to(device)
                
                # Forward prop.
                h1 = encoder(imgs1)
                h2 = encoder(imgs2)
                h  = torch.cat((h1, h2), axis=3)
                
                # "<start>": 8549, "<end>": 8550, "<pad>": 0, ".": 16
                # 将caps1的end置未pad， 将caps2的start置为pad，然后将caps1和caps2的有效部分concat起来，这样便只有一个start和一个end
                caps = torch.zeros(size=(caps1.shape[0], caps1.shape[1]*2), dtype=caps1.dtype).cuda()
                for idx, (cap1, cap2) in enumerate(zip(caps1, caps2)):
                    cap1_end_idx = cap1.tolist().index(8550)
                    cap2_end_idx = cap2.tolist().index(8550)
                    
                    caps[idx][:cap1_end_idx] = cap1[:cap1_end_idx]
                    caps[idx][cap1_end_idx:(cap1_end_idx) + (cap2_end_idx)] = cap2[1:cap2_end_idx+1]
                    
                caplens = caplens1 + caplens2 - 2

                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(h, caps, caplens)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i+1, min(len(vals1), len(vals2)), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
                LOSS.append(loss.item())
                ACCS.append(top5)
        
        else:
            for i, ((imgs1, caps1, caplens1, _), (imgs2, caps2, caplens2, _)) in enumerate(zip(val_loader1, val_loader2)):
                if imgs1.shape[0] != imgs2.shape[0]:
                    break
                # Move to GPU, if available
                imgs1 = imgs1.to(device)
                caps1 = caps1.to(device)
                caplens1 = caplens1.to(device)

                # Move to GPU, if available
                imgs2 = imgs2.to(device)
                caps2 = caps2.to(device)
                caplens2 = caplens2.to(device)
                
                # Forward prop.
                h1 = encoder(imgs1)
                h2 = encoder(imgs2)
                h  = torch.cat((h1, h2), axis=3)
                
                # "<start>": 8549, "<end>": 8550, "<pad>": 0, ".": 16
                # 将caps1的end置未pad， 将caps2的start置为pad，然后将caps1和caps2的有效部分concat起来，这样便只有一个start和一个end
                caps = torch.zeros(size=(caps1.shape[0], caps1.shape[1]*2), dtype=caps1.dtype).cuda()
                for idx, (cap1, cap2) in enumerate(zip(caps1, caps2)):
                    cap1_end_idx = cap1.tolist().index(8550)
                    cap2_end_idx = cap2.tolist().index(8550)
                    
                    caps[idx][:cap1_end_idx] = cap1[:cap1_end_idx]
                    caps[idx][cap1_end_idx:(cap1_end_idx) + (cap2_end_idx)] = cap2[1:cap2_end_idx+1]
                
                caplens = caplens1 + caplens2 - 2

                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(h, caps, caplens)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i+1, min(len(val_loader1), len(val_loader2)), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
                LOSS.append(loss.item())
                ACCS.append(top5)

        return LOSS, ACCS, loss.item(), top5

"""
Training and validation.
"""

# global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name1,data_name2, word_map,data_folder

# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# 断点续训
fine_tune = True    # True:继续训练之前保存的模型   False:重新训练模型
load_epoch = 13      # 加载在第几个epoch保存的模型
models_path = '/workspace/code/SampleNIC/models/'    # 存放模型的目录地址

def load_model(epoch, models_path):  
    encoder_name = 'encoder_saved_epoch_{}'.format(epoch)
    encoder = torch.load(os.path.join(models_path, encoder_name))
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr)  if fine_tune_encoder else None
    
    decoder_name = 'decoder_saved_epoch_{}'.format(epoch)
    decoder = torch.load(os.path.join(models_path, decoder_name))
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
                                         
    return (encoder, encoder_optimizer), (decoder, decoder_optimizer)

if fine_tune:
    (encoder, encoder_optimizer), (decoder, decoder_optimizer) = load_model(load_epoch,models_path)
else:
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
                                         
    encoder = Encoder()
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None
    

# Move to GPU, if available
decoder = decoder.to(device)
encoder = encoder.to(device)

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_loader1 = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name1, 'TRAIN', transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, pin_memory=True)  # flickr

val_loader1 = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name1, 'VAL', transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, pin_memory=True) # flickr

train_loader2 = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name2, 'TRAIN', transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True,  pin_memory=True) # PCCD

val_loader2 = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name2, 'VAL', transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, pin_memory=True) # PCCD

# LOSS & ACCS
train_losses_step = []
train_losses_epoch = []
train_accs_step = []
train_accs_epoch = []

val_losses_step = []
val_losses_epoch = []
val_accs_step = []
val_accs_epoch = []

if __name__ == '__main__':
  for epoch in range(start_epoch, epochs):
  
      # 在第30、60、90个epoch降低学习率, 
      if epoch == 5:
          adjust_learning_rate(decoder_optimizer, 0.8)  # 学习率降低为原来的0.8倍
      elif epoch == 10:
          adjust_learning_rate(decoder_optimizer, 0.8)
      elif epoch == 20:
          adjust_learning_rate(decoder_optimizer, 0.8)
      
      # 再第...个epoch保存模型
      save_epochs = [1, 5, 9, 13,17,21,25,29,33,37,40]  
            
      # One epoch's training
      train_LOSS, train_ACCS, train_losses, train_top5accs = train(train_loader1=train_loader1,
            train_loader2=train_loader2,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch, 
            save_epochs=save_epochs, 
            models_path=models_path)
      train_accs_epoch.append(train_top5accs)
      train_losses_epoch.append(train_losses)
      for each in train_LOSS:
          train_losses_step.append(each)
      for each in train_ACCS:
          train_accs_step.append(each)
  
      # One epoch's validation
      val_LOSS, val_ACCS, val_losses, val_top5accs = validate(val_loader1=val_loader1,
                              val_loader2=val_loader2,
                              encoder=encoder,
                              decoder=decoder,
                              criterion=criterion)
      val_accs_epoch.append(val_top5accs)
      val_losses_epoch.append(val_losses)
      for each in val_LOSS:
          val_losses_step.append(each)
      for each in val_ACCS:
          val_accs_step.append(each)
      
  # Save losses % accs
  np.save('train_losses_step.npy', np.array(train_losses_step))
  np.save('train_losses_epoch.npy', np.array(train_losses_epoch))
  np.save('train_accs_step.npy', np.array(train_accs_step))
  np.save('train_accs_epoch.npy', np.array(train_accs_epoch))
  np.save('val_losses_step.npy', np.array(val_losses_step))
  np.save('val_losses_epoch.npy', np.array(val_losses_epoch))
  np.save('val_accs_step.npy', np.array(val_accs_step))
  np.save('val_accs_epoch.npy', np.array(val_accs_epoch))
  
  
  # Save models
  torch.save(encoder, 'encoder')
  torch.save(decoder, 'decoder')
