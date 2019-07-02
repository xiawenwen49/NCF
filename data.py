import os, logging
import mxnet as mx
from Dataset import *
def get_movielens_data(data_dir, prefix):
    if not os.path.exists(os.path.join(data_dir, "ml-10M100K")):
        mx.test_utils.get_zip_data(data_dir,
                                   "http://files.grouplens.org/datasets/movielens/%s.zip" % prefix,
                                   prefix + ".zip")
        assert os.path.exists(os.path.join(data_dir, "ml-10M100K"))
        os.system("cd data/ml-10M100K; chmod +x allbut.pl; sh split_ratings.sh; cd -;")

def get_movielens_iter(filename, batch_size):
    """Not particularly fast code to parse the text file and load into NDArrays.
    return two data iters, one for train, the other for validation.
    """
    logging.info("Preparing data iterators for " + filename + " ... ")
    user = []
    item = []
    score = []
    with open(filename, 'r') as f:
        num_samples = 0
        for line in f:
            tks = line.strip().split('::')
            if len(tks) != 4:
                continue
            num_samples += 1
            user.append((tks[0]))
            item.append((tks[1]))
            score.append((tks[2]))
    # convert to ndarrays
    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item)
    score = mx.nd.array(score)
    # prepare data iters
    data_train = {'user': user, 'item': item}
    label_train = {'score': score}
    iter_train = mx.io.NDArrayIter(data=data_train,label=label_train,
                                   batch_size=batch_size, shuffle=True)
    return mx.io.PrefetchingIter(iter_train)

def get_train_instances(train, num_negatives): # train是scipy.sparse.dok_matrix类型
    user_input, item_input, labels = [],[],[]
    num_users, num_items = train.shape
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u,j) in train.keys():        
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def get_train_iters(train, num_negatives, batch_size):
    user, item, label = get_train_instances(train, num_negatives)

    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item, dtype='int32')
    label = mx.nd.array(label) 
    
    data_train = {'user': user, 'item': item}
    label_train = {'score': label}
    iter_train = mx.io.NDArrayIter(data=data_train,label=label_train,
                                   batch_size=batch_size, shuffle=True)
    return mx.io.PrefetchingIter(iter_train)

def get_eval_iters(user, item, batch_size):
    data_eval = {'user': user, 'item': item}
    label = np.zeros(len(item))
    iter_eval = mx.io.NDArrayIter(data=data_eval,label=label,
                                   batch_size=batch_size, shuffle=False) #!!!!
    return iter_eval

    