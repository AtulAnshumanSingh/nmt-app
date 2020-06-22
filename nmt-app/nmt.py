import os
from docopt import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from vocab import *
import math
import time

class Encoder(tf.keras.Model):
    
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    mask = tf.math.logical_not(tf.math.equal(x, 0))
    mask = tf.expand_dims(mask, axis = 2)
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    mask = tf.cast(mask, dtype=output.dtype)
    output = output*mask
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class Attention(tf.keras.Model):
    
  def __init__(self, units):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):

    hidden_with_time_axis = tf.expand_dims(query, 1)

    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
    
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    self.attention = Attention(self.dec_units)

  def call(self, x, hidden, enc_output):

    context_vector, attention_weights = self.attention(hidden, enc_output)

    x = self.embedding(x)

    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    output, state = self.gru(x)

    output = tf.reshape(output, (-1, output.shape[2]))

    x = self.fc(output)

    return x, state, attention_weights

class NMT(tf.keras.Model):
    
    def __init__(self, vocab_inp_size, vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE):
        super(NMT, self).__init__()
        self.encoder = Encoder(vocab_inp_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
        self.decoder = Decoder(vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
    
    def call(self, source, target, VOCAB, enc_hidden, loss_object, BATCH_SIZE):
        
        enc_output, enc_hidden = self.encoder(source, enc_hidden)
        dec_hidden = enc_hidden
        
        dec_input = tf.expand_dims([VOCAB.tgt['<s>']] * BATCH_SIZE, 1)
        
        loss = 0
        
        for t in range(1, target.shape[1]):
            
          predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
    
          loss += self.loss_function(target[:, t], predictions, loss_object)
    
          dec_input = tf.expand_dims(target[:, t], 1)
        
        return loss
        
    def loss_function(self,real, pred, loss_object):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = loss_object(real, pred)
    
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
    
      return tf.reduce_mean(loss_)

@tf.function
def train_step(model, inp, targ, VOCAB, enc_hidden, optimizer, loss_object, BATCH_SIZE):
  loss = 0
  
  with tf.GradientTape() as tape:
    loss = model(inp, targ, VOCAB, enc_hidden, loss_object, BATCH_SIZE)

  batch_loss = (loss / int(targ.shape[1]))

  variables = model.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def train(dataset, EMBED_SIZE, HIDDEN_SIZE, DROPOUT_RATE, BATCH_SIZE, NUM_TRAIN_STEPS, BUFFER_SIZE, steps_per_epoch, vocab_inp_size, vocab_tar_size, VOCAB):
    
    print("initializing seq2seq model...")
    model = NMT(vocab_inp_size, vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
    
    print("initializing seq2seq model... optimizer")
    optimizer = tf.keras.optimizers.Adam()
    
    print("initializing seq2seq model... loss")
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    print("initializing seq2seq model... defining checkpoint")
    checkpoint, checkpoint_prefix = define_checkpoints(optimizer, model)
    
    for epoch in range(NUM_TRAIN_STEPS):
        
      start = time.time()
    
      enc_hidden = model.encoder.initialize_hidden_state()
      total_loss = 0
    
      for (batch, (inp, targ)) in enumerate(dataset.take(BUFFER_SIZE)):
    
        batch_loss = train_step(model, inp, targ, VOCAB, enc_hidden, optimizer, loss_object, BATCH_SIZE)
        total_loss += batch_loss
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
      print("Saving checkpoint...")
      checkpoint.save(file_prefix = checkpoint_prefix)
    
      print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / BATCH_SIZE))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
     
    print('saving weights!')
    model.save_weights('nmt_model',save_format='hdf5')

def decode(model, sent_file, vocab, push_to_file=0):
    
    print('processing sentence...')
    VOCAB = vocab
    src_sents = read_file(sent_file, source='src')
    src_pad = VOCAB.src.to_input_tensor(src_sents)
    
    for sents in src_pad:
        
        sents = sents.reshape(1,-1)

        result = ''
        
        hidden = [tf.zeros((1, model.encoder.enc_units))]
        enc_out, enc_hidden = model.encoder(sents, hidden)
        
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([VOCAB.tgt['<s>']], 0)
        
        for t in range(50):
            predictions, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_out)
            
            attention_weights = tf.reshape(attention_weights, (-1, ))
    
            predicted_id = tf.argmax(predictions[0]).numpy()
    
            result += VOCAB.tgt.indices2words([predicted_id])[0] + ' '
    
            if VOCAB.tgt.indices2words([predicted_id])[0] == '</s>':
                break
            
            dec_input = tf.expand_dims([predicted_id], 0)
        
        print('Translation:',  result)

def decode_sentence(model, sent_file, vocab):
    
    print('processing sentence...')
    VOCAB = vocab
    src_sents = read_sent(sent_file, source='src')
    
    print(src_sents)
    src_pad = VOCAB.src.to_input_tensor(src_sents)
    print(src_pad.shape)

    result = ''
    
    hidden = [tf.zeros((1, model.encoder.enc_units))]
    enc_out, enc_hidden = model.encoder(src_pad, hidden)
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([VOCAB.tgt['<s>']], 0)
    
    for t in range(50):
        predictions, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_out)
        
        attention_weights = tf.reshape(attention_weights, (-1, ))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += VOCAB.tgt.indices2words([predicted_id])[0] + ' '

        if VOCAB.tgt.indices2words([predicted_id])[0] == '</s>':
            return result, sent_file
        
        dec_input = tf.expand_dims([predicted_id], 0)
    
    print('Sentence:',  sent_file)
    print('Translation:',  result)
