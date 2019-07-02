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
import logging
import mxnet as mx
from data import get_eval_iters
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
    for idx in range(len(_testRatings)):
        
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        print('\revaluate idx:{}, hr:{}'.format(idx, hr), end='')
    print('\n')
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx] # {user,+item}
    items = _testNegatives[idx] # {-item1,-item2,...,-item99}
    u = rating[0] # 用户
    gtItem = rating[1] # 正例的物品索引
    items.append(gtItem) # 把正例的物品索引加到99个负例的物品索引之后
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32') # 100维的user 
    iter_eval = get_eval_iters(users, items, batch_size=100)
    predictions = _model.predict( iter_eval ) # eval_data must be of type NDArray or DataIter, new_nd_iter
    # predictions = _model.predict(mx.nd.array([users, np.array(items)]) )
    for i in range(len(items)): # 100个标号 
        item = items[i]
        map_item_score[item] = predictions[i] # {item：score}的字典
    items.pop() # 删除最后一个列表值
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get) # 依据score的大小排列，返回前K个item标号
    hr = getHitRatio(ranklist, gtItem) # ranklist是top-K个item标号
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem): # 如果top-K的item标号中包含了实际的正例标号，返回1；否则返回0
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):  # i为实际的正例在top-K的item中的位置
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2) # 实际的正例越靠后，NDCG值越小
    return 0



