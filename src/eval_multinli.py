'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''
import numpy as np
import os.path
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import codecs
from unicode_tr import unicode_tr
import sentencepiece as spm
import json_lines

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

def evaluate(encoder, seed=1234, evaltest=False, loc='./data/', file_meta_data=None, sp=None):
    """
    Run experiment
    """
    
    print 'Preparing data...'
    train, dev, test, labels = load_data(loc, file_meta_data, sp)
    #train[0], train[1], labels[0] = shuffle(train[0], train[1], labels[0], random_state=seed)

    print 'Computing training skipthoughts...'
    trainA = encoder.encode(train[0], verbose=True, use_eos=True, max_sent_len=100)
    trainB = encoder.encode(train[1], verbose=True, use_eos=True, max_sent_len=100)
 
    print 'Computing development skipthoughts...'
    devA = encoder.encode(dev[0], verbose=True, use_eos=True, max_sent_len=100)
    devB = encoder.encode(dev[1], verbose=True, use_eos=True, max_sent_len=100)

    print 'Computing feature combinations...'
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    print 'Encoding labels...'
    trainY, label_encoder = train_label_encoder(labels[0])
    devY = encode_labels(labels[1], label_encoder)
    
    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1], nclass=trainY.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY)

    if evaltest:
        prepare_test_output(encoder, bestlrmodel, test, labels[2], label_encoder, loc, file_meta_data)
        print('done')
        return
    print('done')

def prepare_test_output(encoder, model, test_data, test_pair_idxs, label_encoder, loc, file_meta_data):
       print 'Computing test quickthoughts...'
       testA = encoder.encode(test_data[0], verbose=True, use_eos=True, max_sent_len=100)
       testB = encoder.encode(test_data[1], verbose=True, use_eos=True, max_sent_len=100)

       print 'Computing feature combinations...'
       testF = np.c_[np.abs(testA - testB), testA * testB]

       print 'Evaluating Test Dataset...'
       test_predicted_probs = model.predict(testF)
       test_predicted_class_idxs = np.argmax(test_predicted_probs,axis=1)
       test_predicted_classes = label_encoder.inverse_transform(test_predicted_class_idxs)
       
       test_output_filename = os.path.join(loc, file_meta_data['file_names']['test_output'])
       print 'Writing Test Dataset...', test_output_filename
       with codecs.open(test_output_filename, mode='w', encoding='utf-8') as f:
          f.write('pairID,gold_label\n')
          for pair_idx, predicted_class in zip(test_pair_idxs, test_predicted_classes):
             f.write(pair_idx + ','+ predicted_class + '\n')
          f.close()
def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    inputs = Input(shape=(ninputs,))
    outputs = Dense(units=nclass, activation='softmax')(inputs)
    lrmodel = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=0.001)
    lrmodel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    lrmodel.summary()
    return lrmodel


def train_model(lrmodel, X, Y, devX, devY):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    
    training_history = lrmodel.fit(X, Y, verbose=2, epochs=100, shuffle=False, validation_data=(devX, devY), callbacks=[early_stopping])
    score = training_history.history['acc'][-1]*100
    val_score = training_history.history['val_acc'][-1]*100
    print 'Training accuracy:' + str(score) + ' Validation accuracy:' + str(val_score)
    dev_scores = lrmodel.evaluate(devX, devY)
    score = dev_scores[1]*100

    print 'Dev Accuracy: ' + str(score)
    return lrmodel
    

def train_label_encoder(labels):
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)
    #integer_encoded_labels = integer_encoded_labels.reshape(len(integer_encoded_labels), 1)
    Y = to_categorical(integer_encoded_labels)
    return Y, label_encoder

def encode_labels(labels, label_encoder):
    """
    One hot label encoding
    """
    integer_encoded_labels = label_encoder.transform(labels)
    #integer_encoded_labels = integer_encoded_labels.reshape(len(integer_encoded_labels), 1)
    Y = to_categorical(integer_encoded_labels)
    return Y

def load_data(loc='./data/', file_meta_data=None, sp=None):
    """
    Load the NLI dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]
    print('loc', loc)

    sentence1_key = file_meta_data['sentence_keys']['sentence1']
    sentence2_key = file_meta_data['sentence_keys']['sentence2']

    with codecs.open(os.path.join(loc, file_meta_data['file_names']['train']), mode='rb', encoding='utf-8') as f:
        for item in json_lines.reader(f):
           if item['gold_label'] == '-':
              continue
           trainA.append(encode_sentence(unicode_tr(item[sentence1_key]).lower(), sp))
           trainB.append(encode_sentence(unicode_tr(item[sentence2_key]).lower(), sp))
           trainS.append(item['gold_label'])

    with codecs.open(os.path.join(loc, file_meta_data['file_names']['dev']), mode='rb', encoding='utf-8') as f:
        for item in json_lines.reader(f):
           if item['gold_label'] == '-':
              continue

           devA.append(encode_sentence(unicode_tr(item[sentence1_key]).lower(), sp))
           devB.append(encode_sentence(unicode_tr(item[sentence2_key]).lower(), sp))
           devS.append(item['gold_label'])

    with codecs.open(os.path.join(loc, file_meta_data['file_names']['test']), mode='rb', encoding='utf-8') as f:
        for item in json_lines.reader(f):
           if item['gold_label'] == '-':
              continue
           testA.append(encode_sentence(unicode_tr(item[sentence1_key]).lower(), sp))
           testB.append(encode_sentence(unicode_tr(item[sentence2_key]).lower(), sp))
           testS.append(item['pairID'])
    return [trainA, trainB], [devA, devB], [testA, testB], [trainS, devS, testS]

def encode_sentence(sentence, sp):
  if sp == None:
     return sentence
  encoded_pieces = sp.EncodeAsPieces(sentence)
  sentence = ' '.join(encoded_pieces)
  return sentence
