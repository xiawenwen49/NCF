import mxnet as mx

def gmf_model(factor_size, num_hidden, max_user, max_item, dense):
    # input
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')
    score = mx.sym.Variable('score')
    stype = 'default' if dense else 'row_sparse'
    sparse_grad = not dense
    user_weight = mx.sym.Variable('user_weight', stype=stype)
    item_weight = mx.sym.Variable('item_weight', stype=stype)
    # user feature lookup
    user = mx.sym.Embedding(data=user, weight=user_weight, sparse_grad=sparse_grad,
                            input_dim=max_user, output_dim=factor_size)
    # item feature lookup
    item = mx.sym.Embedding(data=item, weight=item_weight, sparse_grad=sparse_grad,
                            input_dim=max_item, output_dim=factor_size)                
    # elementwise product of user and item embeddings
    pred = user*item
    # Final prediction layer  
    pred = mx.sym.FullyConnected(data=pred,num_hidden=1)
    pred = mx.sym.Activation(data=pred, act_type='sigmoid')
    pred = mx.sym.Flatten(data=pred)  
    # loss layer
    pred = mx.sym.LinearRegressionOutput(data=pred, label=score)
    return pred