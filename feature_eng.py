import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import numpy as np
import time
import math
from hanziconv import HanziConv
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GaussianDropout, Flatten, Merge
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint



def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


#
# song_extra_info = pd.read_csv('data/song_extra_info.csv')


EMBEDDING_DIMENSION = 50
emb = {}
song_info = pd.read_csv('data/songs.csv')
user_info = pd.read_csv('data/members.csv')
train_data = pd.read_csv('data/train.csv')


def str_clean(s):
    s = str(s).lower().strip()
    s = HanziConv.toSimplified(s)
    return s


def build_song_vocab_embedding():
    # song_length = song_info['song_length'].astype('category') # need to divide into ranges

    '''
    =================================================
    Build vocab for genres
    '''
    song_genre = song_info['genre_ids']# need to divide into ranges
    tmp_dict = set()
    for genre in song_genre:
        genre = str(genre)
        split = genre.split('|')
        if len(split) == 1:
            s = split[0]
            if s not in tmp_dict:
                emb['genre_'+s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
                tmp_dict.add(s)
        else:
            for s in split:
                if s not in tmp_dict:
                    emb['genre_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
                    tmp_dict.add(s)
    emb['genre_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)

    '''
    =================================================
    Build vocab for artist_name
    '''
    song_artist_name = song_info['artist_name']  # need to divide into ranges
    tmp_dict = set()
    for s in song_artist_name:
        s = str_clean(s)
        if s not in tmp_dict:
            emb['artist_name_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
            tmp_dict.add(s)
    emb['artist_name_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)

    '''
    =================================================
    Build vocab for composer
    '''
    song_composer = song_info['composer']  # need to divide into ranges
    tmp_dict = set()
    for s in song_composer:
        s = str_clean(s)
        if s not in tmp_dict:
            emb['composer_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
            tmp_dict.add(s)
    emb['composer_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)

    '''
    =================================================
    Build vocab for lyricist
    '''
    song_lyricist = song_info['lyricist']  # need to divide into ranges
    tmp_dict = set()
    for s in song_lyricist:
        s = str_clean(s)
        if s not in tmp_dict:
            emb['lyricist_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
            tmp_dict.add(s)
    emb['lyricist_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)

    '''
    =================================================
    Build vocab for language
    '''
    song_language = song_info['language']  # need to divide into ranges
    tmp_dict = set()
    for s in song_language:
        s = str_clean(s)
        if s not in tmp_dict:
            emb['language_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
            tmp_dict.add(s)
    emb['language_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)


def build_user_vocab_embedding():
    '''
    =================================================
    Build vocab for city
    '''
    user_city = user_info['city']  # need to divide into ranges
    tmp_dict = set()
    for s in user_city:
        s = str_clean(s)
        if s not in tmp_dict:
            emb['city_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
            tmp_dict.add(s)
    emb['city_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)

    '''
    =================================================
    Build vocab for gender
    '''
    user_gender = user_info['gender']  # need to divide into ranges
    tmp_dict = set()
    for s in user_gender:
        s = str_clean(s)
        if s not in tmp_dict:
            emb['gender_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
            tmp_dict.add(s)
    emb['gender_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)

    '''
    =================================================
    Build vocab for registered_via
    '''
    user_registered_via = user_info['registered_via']  # need to divide into ranges
    tmp_dict = set()
    for s in user_registered_via:
        s = str_clean(s)
        if s not in tmp_dict:
            emb['registered_via_' + s] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)
            tmp_dict.add(s)
    emb['registered_via_' + '<UNK>'] = np.random.uniform(-1, 1, EMBEDDING_DIMENSION)


def build_sentence():
    train_sent = []

    print('Building user dictionary...')
    # build dictionary first
    user_info_dict = {}
    for idx, user in user_info.iterrows():
        user_info_dict[user['msno']] = {'city': user['city'], 'gender': user['gender'],
                                        'registered_via': user['registered_via']}
    print('Building song dictionary...')
    song_info_dict = {}
    for idx, song in song_info.iterrows():
        song_info_dict[song['song_id']] = {'artist_name': song['artist_name'], 'genre_ids': song['genre_ids'],
                                           'composer': song['composer'], 'lyricist': song['lyricist'],
                                           'language': song['language']}

    print('Building sentences...')
    start = time.time()
    for idx, meta in train_data.iterrows():
        if idx % 10000 == 0 and idx > 1:
            print(idx/len(train_data), time_since(start, idx / len(train_data)))
        sent = []
        user_id = meta['msno']
        if user_id not in user_info_dict:
            city = 'city_' + '<UNK>'
            gender = 'gender_' + '<UNK>'
            registered_via = 'registered_via_' + '<UNK>'
        else:
            selected_user = user_info_dict[user_id]
            city = 'city_' + str_clean(selected_user['city'])
            gender = 'gender_' + str_clean(selected_user['gender'])
            registered_via = 'registered_via_' + str_clean(selected_user['registered_via'])
        sent.extend([city, gender, registered_via])

        song_id = meta['song_id']
        if song_id not in song_info_dict:
            genre = 'genre_' + '<UNK>'
            artist_name = 'artist_name_' + '<UNK>'
            composer = 'composer_' + '<UNK>'
            lyricist = 'lyricist_' + '<UNK>'
            language = 'language_' + '<UNK>'
            print('Song: ', song_id, ' doest exist')
        else:
            selected_song = song_info_dict[song_id]
            genre = 'genre_' + str_clean(selected_song['genre_ids']).split('|')[0]  # only the first one!
            artist_name = 'artist_name_' + str_clean(selected_song['artist_name'])
            composer = 'composer_' + str_clean(selected_song['composer'])
            lyricist = 'lyricist_' + str_clean(selected_song['lyricist'])
            language = 'language_' + str_clean(selected_song['language'])
        sent.extend([genre, artist_name, composer, lyricist, language])
        train_sent.append(sent)
    import pickle
    with open('train_sent.pkl', 'bw') as f:
        pickle.dump(train_sent, f)
    return train_sent

def train():
    VALIDATION_SPLIT = 0.1
    num_lstm = 300
    rate_drop_lstm = 0.2
    import pickle
    with open('train_sent.pkl', 'br') as f:
        train_sent = pickle.load(f)
    with open('emb.pkl', 'br') as f:
        emb = pickle.load(f)
    NUM_VOCAB = len(emb)
    word2id = {}
    id2word = {}
    idx = 0
    for word in emb:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    embedding_matrix = np.zeros((NUM_VOCAB, EMBEDDING_DIMENSION))
    for i, word in id2word.items():
        embedding_vector = emb[word]
        embedding_matrix[i] = embedding_vector


    train_target = train_data['target'].tolist()
    perm = np.random.permutation(len(train_sent))
    idx_train = perm[:int(len(train_sent) * (1 - VALIDATION_SPLIT))]
    idx_val = perm[int(len(train_sent) * (1 - VALIDATION_SPLIT)):]

    data_train = [train_sent[x] for x in idx_train]
    data_val = [train_sent[x] for x in idx_val]
    label_train = [train_target[x] for x in idx_train]
    label_val = [train_target[x] for x in idx_val]

    data_train = [[word2id[y] for y in x] for x in data_train]
    data_val = [[word2id[y] for y in x] for x in data_val]

    data_train = np.array(data_train)
    data_val = np.array(data_val)

    print(data_train.shape)

    MAX_SEQUENCE_LENGTH = data_train.shape[1]
    embedding_layer = Embedding(NUM_VOCAB,
                                EMBEDDING_DIMENSION,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    lstm_layer = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))


    input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded = embedding_layer(input)
    rnn = lstm_layer(embedded)
    rnn = Dense(300, activation='relu')(rnn)
    preds = Dense(1, activation='sigmoid')(rnn)
    model = Model(inputs=input, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = 'tmp' + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    model.summary()
    hist = model.fit(data_train, label_train,
                     validation_data=(data_val, label_val),
                     epochs=20, batch_size=128, shuffle=True, \
                     callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])

if __name__ == '__main__':
    build_user_vocab_embedding()
    build_song_vocab_embedding()
    import pickle

    with open('emb.pkl', 'bw') as f:
        pickle.dump(emb, f)

    # train_sent = build_sentence()
    train()

