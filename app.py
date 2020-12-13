import sys, os, re, csv, codecs, numpy as np, pandas as pd
import pickle
from flask import Flask, jsonify, render_template, request
from flask_ngrok import run_with_ngrok
from IPython.core.display import display, HTML
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import h5py
from keras.models import load_model

df = pd.read_csv('./data/train.csv', encoding='utf-8')
max_len = 200

tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False)

# Fit and run tokenizer
tokenizer_obj = tokenizer.fit_on_texts(df.loc[:, 'comment_text'].values)

def toxicity_level(message):
    """
    Return toxicity probability based on inputed string.
    """
    # Process string
    new_string = [message]
    new_string = tokenizer_obj.texts_to_sequences(new_string)
    new_string = pad_sequences(new_string, maxlen=max_len, padding='post', truncating='post')
    
    print("call predict")
    # Load in pretrained model
    loaded_model = load_model('./models/my_model.h5')
    print("Loaded model from disk")

    # Predict
    prediction = loaded_model.predict(x=new_string)
    
    result = {}
    
    # Print output
    result['Toxic'] = '{:.0%}'.format(prediction[0][0])
    result['Severe Toxic'] =  '{:.0%}'.format(prediction[0][1])
    result['Obscene'] =      '{:.0%}'.format(prediction[0][2])
    result['Threat'] =        '{:.0%}'.format(prediction[0][3])
    result['Insult'] =       '{:.0%}'.format(prediction[0][4])
    result['Identity Hate'] = '{:.0%}'.format(prediction[0][5])
    
    return result


app = Flask(__name__, template_folder='./')
run_with_ngrok(app)

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        message = request.form['message']
        print(message)
        response =  toxicity_level(message)
        print(type(response))
        return response
    #return jsonify("Input text")

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()