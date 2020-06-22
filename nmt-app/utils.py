import math
from typing import List
import numpy as np 
import tensorflow as tf
import os    

def read_file(filepath, source):

    data = []
    
    for line in open(filepath):
        
        sentence = line.strip().split(' ')
        """ append "<s> and </s> only to target source """
        
        if source == 'tgt':
            sentence = ['<s>'] + sentence + ['</s>']
        
        data.append(sentence)
    
    return data

def read_sent(sent, source):
    
    data = []
        
    sentence = sent.strip().split(' ')
    """ append "<s> and </s> only to target source """
    
    if source == 'tgt':
        sentence = ['<s>'] + sentence + ['</s>']
    
    data.append(sentence)
    
    return data

def pad_sents(sents, pad_token):

    sents_padded = []
    
    max_len = 0
        
    for sent in sents:
        
        if len(sent) > max_len:
            max_len = len(sent)
    
    sent = None
    
    for sent in sents:
        
        if len(sent) < max_len:
            sent = sent + [pad_token] * (max_len - len(sent))
        
        sents_padded.append(sent) 
    
    return sents_padded

def batch_iter(data, batch_size, shuffle=False):

    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def define_checkpoints(optimizer, model):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model = model)
    return checkpoint, checkpoint_prefix
    