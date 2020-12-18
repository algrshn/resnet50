import tensorflow as tf


class Model(tf.Module):
    
    def __init__(self,start_conv=[7,64,2],start_pool=[3,2],structure_of_blocks=[[[1,64],[3,64],[1,256]],[[1,128],[3,128],[1,512]],[[1,256],[3,256],[1,1024]],[[1,512],[3,512],[1,2048]]],lengths_of_blocks=[3,4,6,3],w_identity_shortcuts_shapes=[[1,1,64,256],[2,2,256,512],[2,2,512,1024],[2,2,1024,2048]]):
                        
        self.start_conv=start_conv
        self.start_pool=start_pool
        self.structure_of_blocks=structure_of_blocks
        self.lengths_of_blocks=lengths_of_blocks
                
        self.curr_num_of_record_steps=tf.Variable(0.0, dtype=tf.dtypes.float32)

        #_start denotes a few layers coming before ResNet blocks
        
        start_conv_filter_size=self.start_conv[0]
        start_conv_depth=self.start_conv[1]
        
        self.w_start=tf.Variable(tf.random.normal(shape=[start_conv_filter_size,start_conv_filter_size,3,start_conv_depth],stddev=tf.math.sqrt(tf.cast(2/(start_conv_filter_size*start_conv_filter_size*3),dtype=tf.dtypes.float32))))
        self.b_start=tf.Variable(tf.zeros(shape=[1,1,1,start_conv_depth]))
        self.beta_start=tf.Variable(tf.zeros(shape=[1,1,1,start_conv_depth]))
        self.gamma_start=tf.Variable(tf.ones(shape=[1,1,1,start_conv_depth]))
        self.mu_start=tf.Variable(tf.zeros(shape=[1,1,1,start_conv_depth]))
        self.V_start=tf.Variable(tf.zeros(shape=[1,1,1,start_conv_depth]))
        
        self.w=[]
        self.b=[]
        self.beta=[]
        self.gamma=[]
        self.mu=[]
        self.V=[]
        
        for stage_num in range(len(self.lengths_of_blocks)):
            for block_num in range(lengths_of_blocks[stage_num]):
                for layer_num in range(3):
                    
                    filter_size=self.structure_of_blocks[stage_num][layer_num][0]
                    depth=self.structure_of_blocks[stage_num][layer_num][1]
                    
                    if(stage_num==0 and block_num==0 and layer_num==0):
                        prev_depth=start_conv_depth
                    else:
                        prev_depth=self.w[-1].shape[3]
                    
                    self.w.append(tf.Variable(tf.random.normal(shape=[filter_size,filter_size,prev_depth,depth],stddev=tf.math.sqrt(tf.cast(2/(filter_size*filter_size*prev_depth),dtype=tf.dtypes.float32)))))
                    
                    self.b.append(tf.Variable(tf.zeros(shape=[1,1,1,depth])))
                    self.beta.append(tf.Variable(tf.zeros(shape=[1,1,1,depth])))
                    self.gamma.append(tf.Variable(tf.ones(shape=[1,1,1,depth])))
                    self.mu.append(tf.Variable(tf.zeros(shape=[1,1,1,depth])))
                    self.V.append(tf.Variable(tf.zeros(shape=[1,1,1,depth])))
                    
        self.w_identity_shortcuts=[]
        for curr_shape in w_identity_shortcuts_shapes:

            self.w_identity_shortcuts.append((1/(curr_shape[0]*curr_shape[0]))*tf.Variable(tf.broadcast_to(tf.eye(num_rows=curr_shape[2], num_columns=curr_shape[3]),shape=curr_shape)))
                
        dense_size=structure_of_blocks[-1][-1][-1]
        self.dense_b=tf.Variable(tf.zeros(shape=[1,1000]))
        self.dense_w=tf.Variable(tf.random.uniform(shape=[dense_size,1000],minval=-tf.math.sqrt(tf.cast(6/(dense_size+1000),dtype=tf.dtypes.float32)),maxval=tf.math.sqrt(tf.cast(6/(dense_size+1000),dtype=tf.dtypes.float32))))
        
        
    @tf.function   
    def __call__(self,X,mode='train'):
        
        var_epsilon=0.001

        #_start denotes a few layers coming before ResNet blocks 

        start_conv_stride=self.start_conv[2]
        start_pool_filter_size=self.start_pool[0]
        start_pool_stride=self.start_pool[1]
        
        Z_start=self.b_start+tf.nn.convolution(input=X,filters=self.w_start,strides=[start_conv_stride,start_conv_stride],padding='SAME')
        
        
        if(mode=='train'):
            #M_start from Z_start, mean and variance from Z_start
            M_start=tf.nn.batch_normalization(Z_start,mean=tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[0],variance=tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[1],offset=self.beta_start,scale=self.gamma_start,variance_epsilon=var_epsilon)
        elif(mode=='inference'):
            #M_start from Z_start, mean=mu, variance=V
            M_start=tf.nn.batch_normalization(Z_start,mean=self.mu_start,variance=self.V_start,offset=self.beta_start,scale=self.gamma_start,variance_epsilon=var_epsilon)
        elif(mode=='calc_bn_avgs'):
            #M_start from Z_start, mean and variance from Z_start
            #mu_start=running average
            #V_start=running average
            M_start=tf.nn.batch_normalization(Z_start,mean=tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[0],variance=tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[1],offset=self.beta_start,scale=self.gamma_start,variance_epsilon=var_epsilon)
            self.mu_start.assign((self.curr_num_of_record_steps*self.mu_start+tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[0])/(self.curr_num_of_record_steps+1))
            self.V_start.assign((self.curr_num_of_record_steps*self.V_start+tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[1])/(self.curr_num_of_record_steps+1))
        
        A_start=tf.nn.elu(M_start)
        P_start=tf.nn.pool(A_start,window_shape=[start_pool_filter_size,start_pool_filter_size],pooling_type='MAX',strides=[start_pool_stride,start_pool_stride],padding="SAME")
        
        L=[]
        
        L.append(P_start)
        
        num=0
        for stage_num in range(len(self.lengths_of_blocks)):
            for block_num in range(self.lengths_of_blocks[stage_num]):
                for layer_num in range(3):
                    
                    #convolution sublayer
                    
                    if(stage_num!=0 and block_num==0 and layer_num==0):
                        L.append(self.b[num]+tf.nn.convolution(input=L[-1],filters=self.w[num],strides=[2,2],padding='SAME'))
                    else:
                        L.append(self.b[num]+tf.nn.convolution(input=L[-1],filters=self.w[num],padding='SAME'))
                        
                    #BN sublayer                    
                    if(mode=='train'):
                        #L from L[-1], mean and variance from L[-1]
                        L.append(tf.nn.batch_normalization(L[-1],mean=tf.nn.moments(L[-1],axes=[0,1,2],keepdims=True)[0], variance=tf.nn.moments(L[-1], axes=[0,1,2], keepdims=True)[1], offset=self.beta[num], scale=self.gamma[num], variance_epsilon=var_epsilon))
                    elif(mode=='inference'):
                        #L from L[-1], mean=mu, variance=V
                        L.append(tf.nn.batch_normalization(L[-1],mean=self.mu[num],variance=self.V[num],offset=self.beta[num],scale=self.gamma[num],variance_epsilon=var_epsilon))
                    elif(mode=='calc_bn_avgs'):
                        #mu=running average
                        #V=running average
                        #L from L[-1], mean and variance from L[-1]
                        self.mu[num].assign((self.curr_num_of_record_steps*self.mu[num]+tf.nn.moments(L[-1],axes=[0,1,2], keepdims=True)[0])/(self.curr_num_of_record_steps+1))
                        self.V[num].assign((self.curr_num_of_record_steps*self.V[num]+tf.nn.moments(L[-1],axes=[0,1,2],keepdims=True)[1])/(self.curr_num_of_record_steps+1)) 
                        L.append(tf.nn.batch_normalization(L[-1],mean=tf.nn.moments(L[-1], axes=[0,1,2],keepdims=True)[0], variance=tf.nn.moments(L[-1], axes=[0,1,2], keepdims=True)[1], offset=self.beta[num], scale=self.gamma[num], variance_epsilon=var_epsilon))
   
                    #Non-linearity sublayer 
                    
                    if(layer_num==2): #identity shortcut used 
                        if(block_num==0): #identity shortcut with transformation
                            
                            L.append(tf.nn.elu(L[-1]+tf.nn.convolution(input=L[-9], filters=self.w_identity_shortcuts[stage_num],strides=[self.w_identity_shortcuts[stage_num].shape[0],self.w_identity_shortcuts[stage_num].shape[0]],padding='SAME')))
                        else:   #identity shortcut w/o transformation                            
                            L.append(tf.nn.elu(L[-1]+L[-9]))
                    else: #no identity shortcut
                        L.append(tf.nn.elu(L[-1]))
                        
                    num+=1
    
        P=tf.reshape(tf.nn.pool(L[-1],window_shape=[7,7],pooling_type='AVG',strides=[7,7],padding="SAME"), shape=[L[-1].shape[0],L[-1].shape[3]])
        
        Z=self.dense_b+tf.matmul(P,self.dense_w)
        A=tf.nn.softmax(Z)
        
        if(mode=='calc_bn_avgs'):
            self.curr_num_of_record_steps.assign_add(1.0)    
        
        return A
    
    @tf.function
    def loss(self,X,Y):
                        
        N=len(X)
        A=self.__call__(X)
        YTA=tf.matmul(tf.transpose(Y),tf.math.log(A+0.00001))
        J=(-1/N)*tf.linalg.trace(YTA)
        
        return J
    
    @tf.function    
    def train_step(self,opt,X,Y):

        current_loss=lambda: self.loss(X,Y)
        varslist = self.b + self.w + self.beta + self.gamma
        
        varslist.append(self.w_start)
        varslist.append(self.b_start)
        varslist.append(self.beta_start)
        varslist.append(self.gamma_start)
        
        varslist.append(self.dense_b)
        varslist.append(self.dense_w)
        
        opt.minimize(current_loss, varslist)
        
        
