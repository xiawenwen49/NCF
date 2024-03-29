# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
# from mxnet.test_utils import *
# from config import *
from MLP import mlp_model
from data import get_movielens_iter, get_movielens_data
import argparse
import os
import time

MOVIELENS = {
    'dataset': 'ml-10m',
    'train': './data/ml-10M100K/r1.train',
    'val': './data/ml-10M100K/r1.test',
    'max_user': 71569,
    'max_movie': 65135,
}

parser = argparse.ArgumentParser(description="Run sparse wide and deep inference",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-infer-batch', type=int, default=100,
                    help='number of batches to inference')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='loading the params of the corresponding training epoch.')
parser.add_argument('--batch-size', type=int, default=100,
                    help='number of examples per batch')
parser.add_argument('--benchmark', action='store_true', default=False,
                    help='run the script for benchmark mode, not set for accuracy test.')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='accurcy for each batch will be logged if set')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Inference on GPU with CUDA')
parser.add_argument('--model-prefix', type=str, default='checkpoint',
                    help='the model prefix')

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_iters = args.num_infer_batch
    batch_size = args.batch_size
    benchmark = args.benchmark
    verbose = args.verbose
    model_prefix = args.model_prefix
    load_epoch = args.load_epoch
    ctx = mx.gpu(0) if args.gpu else mx.cpu()

    # data iterator
    data_dir = os.path.join(os.getcwd(), 'data')
    get_movielens_data(data_dir, MOVIELENS['dataset'])
    eval_data = get_movielens_iter(MOVIELENS['val'], batch_size)
    
    # load parameters and symbol
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, load_epoch)

    # module
    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['user', 'item'], label_names=['score']) 
                        
    mod.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label)
    # get the sparse weight parameter
    mod.set_params(arg_params=arg_params, aux_params=aux_params)

    data_iter = iter(eval_data)
    nbatch = 0
    if benchmark:
        logging.info('Inference benchmark started ...')
        tic = time.time()
        for i in range(num_iters):
            try:
                batch = data_iter.next()
            except StopIteration:
                data_iter.reset()
            else:
                mod.model_prefix(batch, is_train=False)
                for output in mod.get_outputs():
                    output.wait_to_read()
                nbatch += 1
        score = (nbatch*batch_size)/(time.time() - tic)
        logging.info('batch size %d, process %s samples/s' % (batch_size, score))
    else:
        logging.info('Inference started ...')
        # use accuracy as the metric
        metric = mx.metric.create(['acc'])
        accuracy_avg = 0.0
        for batch in data_iter:
            nbatch += 1
            metric.reset()
            mod.forward(batch, is_train=False)
            mod.update_metric(metric, batch.label)
            # print(metric.get())
            # print(metric.get() [0][1])
            accuracy_avg += metric.get()[1][0]
            if args.verbose:
                logging.info('batch %d, accuracy = %s' % (nbatch, metric.get()))
        logging.info('averged accuracy on eval set is %.5f' % (accuracy_avg/nbatch))