'''
Created on Aug 9, 2016

@author: he8819197
'''
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model, F1_score
from Dataset import Dataset
from time import time
import sys
import GMFlogistic, MLPlogistic

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0, enable_dropout=False):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = "mlp_embedding_user",
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'mlp_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   

    # Auxiliary input of users and items
    aux_user_input = Input(shape=(4,), name='aux_user_input')
    mf_latent_aux_user_input = Dense(5, name='mf_latent_aux_user_input')(aux_user_input)
    mlp_latent_aux_user_input = Dense(5, name='mlp_latent_aux_user_input')(aux_user_input)
    mf_full_user_input = merge([MF_Embedding_User(user_input), mf_latent_aux_user_input], mode='concat')
    mlp_full_user_input = merge([MLP_Embedding_User(user_input), mlp_latent_aux_user_input], mode='concat')

    aux_item_input = Input(shape=(2,), name='aux_item_input')
    mf_latent_aux_item_input = Dense(5, name='mf_latent_aux_item_input')(aux_item_input)
    mlp_latent_aux_item_input = Dense(5, name='mlp_latent_aux_item_input')(aux_item_input)
    mf_full_item_input = merge([MF_Embedding_Item(item_input), mf_latent_aux_item_input], mode='concat')
    mlp_full_item_input = merge([MLP_Embedding_Item(item_input), mlp_latent_aux_item_input], mode='concat')

    # MF part
    mf_user_latent = Flatten()(mf_full_user_input)
    mf_item_latent = Flatten()(mf_full_item_input)
    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(mlp_full_user_input)
    mlp_item_latent = Flatten()(mlp_full_item_input)
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    if enable_dropout:
        predict_vector = Dropout(0.5)(predict_vector)
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[[user_input, aux_user_input], [item_input, aux_item_input]],
                  output=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MF lantent auxiliary input
    gmf_user_latent_aux_input = gmf_model.get_layer('latent_aux_user_input').get_weights()
    gmf_item_latent_aux_input = gmf_model.get_layer('latent_aux_item_input').get_weights()
    model.get_layer('mf_latent_aux_user_input').set_weights(gmf_user_latent_aux_input)
    model.get_layer('mf_latent_aux_item_input').set_weights(gmf_item_latent_aux_input)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP lantent auxiliary input
    mlp_user_latent_aux_input = gmf_model.get_layer('latent_aux_user_input').get_weights()
    mlp_item_latent_aux_input = gmf_model.get_layer('latent_aux_item_input').get_weights()
    model.get_layer('mlp_latent_aux_user_input').set_weights(mlp_user_latent_aux_input)
    model.get_layer('mlp_latent_aux_item_input').set_weights(mlp_item_latent_aux_input)
    
    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

"""def get_train_instances(train, num_negatives, weight_negatives, user_weights):
    user_input, item_input, labels, weights = [],[],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        weights.append(user_weights[u])
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            weights.append(weight_negatives * user_weights[u])
    return user_input, item_input, labels, weights"""
def get_train_instances(mini_batch, num_negatives):
    df_products = pd.read_csv('dataset/products.csv')
    df_products = df_products.drop('product_name', 1)
    arr_products = np.array(df_products.index)
    print arr_products
    order_product = mini_batch[:, [0, 6]]
    order_set = list(set(order_product[:, 0]))
    for order_id in order_set:
        arr_i_pid = np.where(order_product[:, 0]==order_id)[0]
        arr_pid = order_product[:, 1][arr_i_pid] - 1
        other_products = np.delete(arr_products, arr_pid) + 1
        negative_products = np.random.choice(other_products, size=num_negatives*len(arr_i_pid), replace=False)
        negative_products = df_products.loc[df_products['product_id'].isin(negative_products.tolist())].values
        negative_samples = np.repeat(np.array([mini_batch[arr_i_pid[0]]]), num_negatives*len(arr_i_pid), axis=0)
        negative_samples[:, [6,7,8]] = negative_products
        negative_samples[:, 9] *= 0
        mini_batch = np.concatenate((mini_batch, negative_samples), axis=0)
    user_input = np.array(mini_batch[:, 1], dtype='int32')
    aux_user_input = mini_batch[:, [2,3,4,5]]
    item_input = np.array(mini_batch[:, 6], dtype='int32')
    aux_item_input = mini_batch[:, [7,8]]
    labels = mini_batch[:, 9]
    return (user_input, aux_user_input, item_input, aux_item_input, labels)



if __name__ == '__main__':
    """dataset = Dataset('dataset')
    fp = open('dataset/prior.csv')
    fp.readline()
    mini_batch = dataset.load_mini_batch(fp, 200)
    print mini_batch.shape
    (user_input, aux_user_input, item_input, aux_item_input, labels) = get_train_instances(mini_batch, 4)
    print user_input.shape
    print aux_user_input.shape
    print item_input.shape
    print aux_item_input.shape
    print labels.shape"""
    #dataset_name = "ml-1m"
    mf_dim = 8    #embedding size
    layers = eval("[16,8]")
    reg_layers = eval("[0,0]")
    reg_mf = 0
    num_negatives = 4   #number of negatives per positive instance
    weight_negatives = 1.0
    learner = "Adam"
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 256
    verbose = 1
    enable_dropout = False
    mf_pretrain = ''
    mlp_pretrain = ''
    
    if (len(sys.argv) > 3):
        dataset_name = sys.argv[1]
        mf_dim = int(sys.argv[2])
        layers = eval(sys.argv[3])
        reg_layers = eval(sys.argv[4])
        reg_mf = float(sys.argv[5])
        num_negatives = int(sys.argv[6])
        weight_negatives = float(sys.argv[7])
        learner = sys.argv[8]
        learning_rate = float(sys.argv[9])
        num_epochs = int(sys.argv[10])
        batch_size = int(sys.argv[11])
        verbose = int(sys.argv[12])
        if (sys.argv[13] == 'true' or sys.argv[13] == 'True'):
            enable_dropout = True
        if len(sys.argv) > 14:
            mf_pretrain = sys.argv[14]
            mlp_pretrain = sys.argv[15]
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF(%s) Dropout %s: mf_dim=%d, layers=%s, regs=%s, reg_mf=%.1e, num_negatives=%d, weight_negatives=%.2f, learning_rate=%.1e, num_epochs=%d, batch_size=%d, verbose=%d"
          %(learner, enable_dropout, mf_dim, layers, reg_layers, reg_mf, num_negatives, weight_negatives, learning_rate, num_epochs, batch_size, verbose))
        
    # Loading data
    t1 = time()
    dataset = Dataset("dataset")
    num_users = dataset.get_num_users()
    num_items = dataset.get_num_items()
    test_set, users, items = dataset.get_user_item_features_test()
    
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, enable_dropout)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMFlogistic.get_model(num_users,num_items,mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLPlogistic.get_model(num_users,num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))
        
    # Init performance
    #(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    #hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    #if hr > 0.6:
    #    model.save_weights('Pretrain/%s_NeuMF_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, layers[-1], num_negatives, hr, ndcg), overwrite=True)
    score = F1_score(model, test_set, users, items, 0.5)
    print('Init: F1_score = %.4f' %score)

    # Training model
    #best_hr, best_ndcg  = hr, ndcg
    for epoch in xrange(num_epochs):
        fp = open('dataset/prior.csv')
        fp.readline()
        i = 0
        while True:
            i += 1
            mini_batch = dataset.load_mini_batch(fp, 200)
            if len(mini_batch) < 5:
                break
            t1 = time()
            # Generate training instances
            user_input, aux_user_input, item_input, aux_item_input, labels = get_train_instances(mini_batch, num_negatives)
            del mini_batch
        
            # Training
            hist = model.fit([[user_input, aux_user_input], [item_input, aux_item_input]], #input
                         np.array(labels), # labels 
                         #sample_weight=np.array(weights), # weight of samples
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
            t2 = time()
        
            # Evaluation
            if i%verbose == 0:
                score = F1_score(model, test_set, users, items, 0.5)
                loss = hist.history['loss'][0]
                print('pass_no %d, mini_batch %d: F1_score = %.4f, loss = %.4f, time %.1f s' %(epoch, i, score, loss, t2-t1))
        fp.close()
    model.save_weights('Training/weights.h5')
    print 'Finish training'
    #print("End. best HR = %.4f, best NDCG = %.4f" %(best_hr, best_ndcg))
    
    """train, testRatings = dataset.trainMatrix, dataset.testRatings
    num_users, num_items = train.shape
    total_weight_per_user = train.nnz / float(num_users)
    train_csr, user_weights = train.tocsr(), []
    for u in xrange(num_users):
        #user_weights.append(total_weight_per_user / float(train_csr.getrow(u).nnz))
        user_weights.append(1)
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))"""
    
