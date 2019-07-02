import mxnet as mx

def mlp_model(factor_size, model_layers, max_user, max_item, dense):
    # input
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')
    score = mx.sym.Variable('score')
    stype = 'default' if dense else 'row_sparse'
    sparse_grad = not dense # default:true
    user_weight = mx.sym.Variable('user_weight', stype=stype)
    item_weight = mx.sym.Variable('item_weight', stype=stype)
    
    # 这里的embedding相当于将one hot的user映射为latent vector（size为factor_size），这个one hot的user是用integer来表示，而非真正的one hot vector
    mlp_user_latent = mx.sym.Embedding(data=user, weight=user_weight, sparse_grad=sparse_grad,
                            input_dim=max_user, output_dim=factor_size)
    mlp_item_latent = mx.sym.Embedding(data=item, weight=item_weight, sparse_grad=sparse_grad,
                            input_dim=max_item, output_dim=factor_size)
    
    # mlp_user_latent = mx.sym.Embedding(data=user, input_dim=max_user, output_dim=factor_size)
    # mlp_item_latent = mx.sym.Embedding(data=item, input_dim=max_item, output_dim=factor_size)
                            

    # Concatenation of two latent features
    # mlp_user_latent = mx.sym.Flatten(mlp_user_latent)
    # mlp_item_latent = mx.sym.Flatten(mlp_item_latent)
    mlp_vector=mx.sym.concat(mlp_user_latent, mlp_item_latent, dim=1) #横轴上的merge 如[[1,2],[3,4]] [[5,6],[7,8]]->[[1,2,5,6],[3,4,7,8]]
    # mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

   
    num_layer = len(model_layers)  # Number of layers in the MLP, model_layers是一个参数
    for layer in range(num_layer):
        mlp_vector = mx.sym.FullyConnected(data=mlp_vector, num_hidden=model_layers[layer])
        mlp_vector=mx.sym.Activation(data=mlp_vector, act_type='relu')
        
        '''
        model_layer = tf.keras.layers.Dense(
            model_layers[layer],
            kernel_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[layer]),
            activation="relu")
        mlp_vector = model_layer(mlp_vector)
        '''
   
    # Final prediction layer
    pred = mx.sym.FullyConnected(data=mlp_vector, num_hidden=1) 
    # pred = mx.sym.Activation(data=pred, act_type='sigmoid')
    '''
    logits = tf.keras.layers.Dense(
      1, activation=None, kernel_initializer="lecun_uniform",
      name=movielens.RATING_COLUMN)(predict_vector)
    '''
    pred = mx.sym.Flatten(data=pred)
    # loss layer
    # pred = mx.sym.LinearRegressionOutput(data=pred, label=score)
    pred = mx.sym.LogisticRegressionOutput(data=pred, label=score)
    return pred
    