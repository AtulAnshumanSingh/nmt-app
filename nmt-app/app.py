import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from vocab import *
from nmt import Encoder, Decoder, Attention, NMT, define_checkpoints, train_step, train, decode, decode_sentence
import math
import time
from flask import Flask, render_template, request


app = Flask(__name__,  template_folder='templates')

@app.route('/', methods = ['GET'])
def home():
   return render_template('home.html')

@app.route('/', methods = ['POST' ,'GET'])
def nmt():

    if request.method == 'POST':
        text = request.form['sent']

        EMBED_SIZE = 256

        HIDDEN_SIZE = 512


        DROPOUT_RATE = 0.2

        BATCH_SIZE = 256

        NUM_TRAIN_STEPS = 10

        VOCAB = Vocab.load('VOCAB_FILE')

        vocab_inp_size = len(VOCAB.src) +1
        vocab_tar_size = len(VOCAB.tgt) +1        

        model = NMT(vocab_inp_size, vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
        sample_hidden = model.encoder.initialize_hidden_state()
        sample_output, sample_hidden = model.encoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden)
        sample_decoder_output, _, _ = model.decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)
        model.load_weights('es_en')

        pred = decode_sentence(model, text, VOCAB)

        return render_template('home.html', result = pred)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
