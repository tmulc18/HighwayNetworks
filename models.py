import tensorflow as tf
from modules import dense_layer, highway_block

class HighwayNetwork(object):
	def __init__(self,num_blocks,num_nodes,input_size,num_classes):
		self.num_blocks = num_blocks
		self.num_nodes = num_nodes
		self.input_size = input_size
		self.num_classes = num_classes

	def initialize(self):
		self.g = tf.Graph()
		with self.g.as_default():
		    self.x = tf.placeholder(tf.float32,shape=[None,self.input_size[-1]])
		    self.y = tf.placeholder(tf.float32,shape=[None,self.num_classes])
		    
		    
		    hn = dense_layer(self.x,num_nodes=self.num_nodes, activation=None, name_suffix='pjct')
		    for i in range(self.num_blocks):
		        hn = highway_block(hn,layer=i,num_nodes=self.num_nodes)
		    
		    self.fc = dense_layer(hn,num_nodes=self.num_classes,activation=None,name_suffix='fc')
		    
		    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc,labels=self.y))
		    
		    self.opt = tf.train.AdamOptimizer(learning_rate=.0001).minimize(self.loss)
		    
		    self.pred = tf.nn.softmax(self.fc)
		    
		    self.init = tf.global_variables_initializer()

	def get_transform_biases(self):
		transform_biases = []
		for v in self.g.get_collection('trainable_variables'):
		    if 'highway' in v.name and 'b_t' in v.name:
		        transform_biases.append(v)
		return transform_biases

	def get_transform_activation(self):
		transform_activations = []
		for op in self.g.get_operations():
		    if 'highway' and 'Sigmoid' in op.name and 'gradient' not in op.name:
		        transform_activations.append(op.values())


class PlainNetwork(object):
	def __init__(self,num_layers,num_nodes,input_size,num_classes):
		self.num_layers = num_layers
		self.num_nodes = num_nodes
		self.input_size = input_size
		self.num_classes = num_classes

	def initialize(self):
		self.g = tf.Graph()
		with self.g.as_default():
		    self.x = tf.placeholder(tf.float32,shape=[None,self.input_size[-1]])
		    self.y = tf.placeholder(tf.float32,shape=[None,self.num_classes])
		    
		    
		    h = dense_layer(self.x,num_nodes=self.num_nodes, activation=None, name_suffix='pjct')
		    for i in range(self.num_layers):
		        h = dense_layer(h,name_suffix='_%d'%i,num_nodes=self.num_nodes)
		    
		    self.fc = dense_layer(h,num_nodes=self.num_classes,activation=None,name_suffix='fc')
		    
		    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc,labels=self.y))
		    
		    self.opt = tf.train.AdamOptimizer(learning_rate=.0001).minimize(self.loss)
		    
		    self.pred = tf.nn.softmax(self.fc)
		    
		    self.init = tf.global_variables_initializer()
