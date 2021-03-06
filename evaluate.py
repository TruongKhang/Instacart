'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import scipy.sparse as sp
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in xrange(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)
    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def F1_score(model, test_set, users_features, items_features, thresh):
    num_user = len(users_features)
    num_item = len(items_features)
    """user_ids = users[0]
    aux_user_input = users[1]
    test_item_ids = test_set[1][0]
    item_ids = items[0]
    aux_item_input = items[1]"""
    matrix_score = sp.dok_matrix((num_user, num_item), dtype='float32')
    for u in range(num_user):
        #user_input = np.repeat([user_ids[u]], len(item_ids))
        #arr_aux = np.repeat([aux_user_input[u]], len(item_ids), axis=0)
        user_u = np.repeat([users_features[u]], num_item, axis=0)
        arr_target = model.predict([user_u[:,0], user_u[:, 1:], 
                                    items_features[:, 0], items_features[:, 1:]])
        indexs = np.where(arr_target>thresh)[0]
        matrix_score[u, indexs] = 1.
        """for i in range(len(item_ids)):
            target_value = model.predict([[user_ids[u], aux_user_input[u]],[item_ids[i], aux_item_input[i]]])
            if target_value > thresh:
                matrix_score[u,i] = 1"""


    arr_index = matrix_score.keys()
    N = len(arr_index)
    true_pos = 0
    for (u,i) in arr_index:
        #indexu_test = np.where(test_user_ids==(user_ids[u]+1))[0]
        vec_inter = np.append(users_features[u], items_features[i])
        find = np.flatnonzero((test_set==vec_inter).all(1))
        if len(find) > 0:
            true_pos += 1
        """for index in indexu_test:
            if np.array_equal(test_aux_user_input[index], aux_user_input[indexu[j]]) is True:
                if test_item_ids[index] == item_ids[indexi[j]]:
                    true_pos += 1"""
    if true_pos == 0:
        return 0
    #print 'true_pos: ', true_pos
    #print 'N: ', N
    #print 'length: ', len(test_set)
    # calculate precision
    precision = float(true_pos)/N
    # calculate recall
    recall = float(true_pos)/len(test_set)
    return 2*precision*recall / (precision+recall)
