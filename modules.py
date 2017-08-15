import tensorflow as tf

def dense_layer(x,num_nodes,activation='relu',name_suffix='',bias_init_val=1):
    """
    Dense fully connected layer
    y = activation(xW+b)
    
    Inputs
    ------------------
    x : input tensor
    
    num_nodes : number of nodes in the layer
    
    acitvation (optional: activation function to use 
         one of ['relu','sigmoid',None]
         default is 'relu'
    
    name_suffix (optional) : the suffix to append to variable names
    
    bias_init_val (optional) : the initial value of the bias

    
    Outputs
    ---------------------
    y : the output tensor
    """
    input_shape = x.get_shape()
    W=tf.get_variable('W'+name_suffix,initializer=tf.random_normal(stddev=.01,shape=[int(input_shape[-1]),num_nodes]))
    b=tf.get_variable('b'+name_suffix,initializer=tf.constant(bias_init_val,shape = [num_nodes],dtype=tf.float32))
    
    logits = tf.matmul(x,W)+b
    
    if activation == None:
        y = logits
    elif activation == 'relu':
        y = tf.nn.relu(logits)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(logits)
    else:
        raise ValueError("Enter a valid activation function")
    
    return y

def highway_block(x,layer,num_nodes=784):
    """
    Highway Network block
    
    H(x)*T(x)+(1-T(x))*x
    
    Input:
    x : tensor input
    
    layer : layer number for the highway block
    
    Output:
    r : last layer in the highway block
    """
    input_shape = x.get_shape()
    with tf.variable_scope('highway_%d'%layer):
        H = dense_layer(x,name_suffix='_h',num_nodes=num_nodes,activation='relu')
        T = dense_layer(x,name_suffix='_t',num_nodes=num_nodes,activation='sigmoid',bias_init_val=-1)
    return H*T + (1.-T)*x