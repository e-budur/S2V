'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''
import numpy as np
import os.path
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import codecs
from unicode_tr import unicode_tr
import sentencepiece as spm
import json_lines

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def evaluate(encoder, seed=1234, evaltest=False, loc='./data/', file_meta_data=None, sp=None):
    """
    Run experiment
    """
    print 'Preparing data...'
    train, dev, test, scores = load_data(loc, file_meta_data, sp)
    train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)
    
    print 'Computing training skipthoughts...'
    trainA = encoder.encode(train[0], verbose=True, use_eos=True)
    trainB = encoder.encode(train[1], verbose=True, use_eos=True)
    
    print 'Computing development skipthoughts...'
    devA = encoder.encode(dev[0], verbose=False, use_eos=True)
    devB = encoder.encode(dev[1], verbose=False, use_eos=True)

    print 'Computing feature combinations...'
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    print 'Encoding labels...'
    trainY = encode_labels(scores[0], file_meta_data)
    devY = encode_labels(scores[1], file_meta_data)
    testY = encode_labels(scores[2], file_meta_data)

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1], nclass=trainY.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[1])

    if evaltest:
        print 'Computing test quickthoughts...'
        testA = encoder.encode(test[0], verbose=False, use_eos=True)
        testB = encoder.encode(test[1], verbose=False, use_eos=True)

        print 'Computing feature combinations...'
        testF = np.c_[np.abs(testA - testB), testA * testB]

        print 'Evaluating Test Dataset...'
        evaluation_scores = bestlrmodel.evaluate(testF, testY)
        print("%s: %.2f%%" % (bestlrmodel.metrics_names[1], evaluation_scores[1]*100))
        return


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(input_dim=ninputs, output_dim=nclass))
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lrmodel


def train_model(lrmodel, X, Y, devX, devY, devscores):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    
    while not done:
        # Every 100 epochs, check Pearson on development set
        training_history = lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
        score = training_history.history['acc'][-1]*100
        val_score = training_history.history['val_acc'][-1]*100
        if score > best:
            print 'Training accuracy:' + str(score) + ' Validation accuracy:' + str(val_score)
            best = score
            bestlrmodel = prepare_model(ninputs=X.shape[1], nclass=Y.shape[1])
            bestlrmodel.set_weights(lrmodel.get_weights())
        else:
            done = True

    dev_scores = bestlrmodel.evaluate(devX, devY)
    score = dev_scores[1]*100

    print 'Dev Accuracy: ' + str(score)
    return bestlrmodel
    

def encode_labels(labels, file_meta_data):
    """
    One hot label encoding
    """
    label_classes = np.array(file_meta_data['label_classes'])
    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)
    integer_encoded_labels = label_encoder.transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_labels = integer_encoded_labels.reshape(len(integer_encoded_labels), 1)
    Y = onehot_encoder.fit_transform(integer_encoded_labels)
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
           testS.append(item['gold_label'])

    return [trainA, trainB], [devA, devB], [testA, testB], [trainS, devS, testS]

def encode_sentence(sentence, sp):
  if sp == None:
     return sentence
  encoded_pieces = sp.EncodeAsPieces(sentence)
  sentence = ' '.join(encoded_pieces)
  return sentence
