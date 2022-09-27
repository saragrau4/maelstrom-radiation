import tensorflow as tf

class BiasMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the mean bias
    """
    def __init__(self, shape, name='bias', **kwargs):
        super(BiasMetric,self).__init__(name=name,**kwargs) # handles base args (e.g., dtype)
        self.shape=shape
        self.total_cm = self.add_weight("total", shape=self.shape, initializer="zeros")
        self.total_batches = self.add_weight("batch", shape=[1], initializer="zeros")
        
    def update_state(self, y_true, y_pred,sample_weight=None):
        y_true_s = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        shape = y_true.shape[0]
        if shape is None:
            shape = 512
        self.total_cm.assign_add(tf.math.reduce_sum((y_true_s-y_pred),axis=0))
        self.total_batches.assign_add(tf.convert_to_tensor([shape],dtype='float32'))
        # self.total_batches.assign_add(shape)
        #return self.total_cm
        
    def result(self):
        #print(self.total_batches.numpy())
        return self.total_cm/self.total_batches
    
    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

class VectorMSE(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the vector mse
    """
    def __init__(self, shape, name='vmse', **kwargs):
        super(VectorMSE,self).__init__(name=name,**kwargs) # handles base args (e.g., dtype)
        self.shape=shape
        self.total_cm = self.add_weight("total", shape=self.shape, initializer="zeros")
        self.total_batches = self.add_weight("batch", shape=[1], initializer="zeros")
        
    def update_state(self, y_true, y_pred,sample_weight=None):
        y_true_s = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        shape = y_true.shape[0]
        if shape is None:
            shape = 512
        self.total_cm.assign_add(tf.math.reduce_sum(tf.math.square(y_true_s-y_pred),axis=0))
        self.total_batches.assign_add(tf.convert_to_tensor([shape],dtype='float32'))
        # self.total_batches.assign_add(shape)
        #return self.total_cm
        
    def result(self):
        #print(self.total_batches.numpy())
        return self.total_cm/self.total_batches
    
    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
