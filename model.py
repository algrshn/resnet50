import tensorflow as tf
import configparser
import sys


class Model(tf.Module):
    
    def __init__(self,start_conv=[7,64,2],start_pool=[3,2],structure_of_blocks=[[[1,64],[3,64],[1,256]],[[1,128],[3,128],[1,512]],[[1,256],[3,256],[1,1024]],[[1,512],[3,512],[1,2048]]],lengths_of_blocks=[3,4,6,3],w_identity_shortcuts_shapes=[[1,1,64,256],[2,2,256,512],[2,2,512,1024],[2,2,1024,2048]]):
        
        #------start reading from config.txt----------------------
        config = configparser.ConfigParser()
        config.read('config.txt')    
        try:
            self.moving_avgs_momentum=float(config.get('batch_renorm','moving_avgs_momentum'))
            self.rmax_max=float(config.get('batch_renorm','rmax_max'))
            self.dmax_max=float(config.get('batch_renorm','dmax_max'))
            self.start_relaxing_rmax=int(config.get('batch_renorm','start_relaxing_rmax'))
            self.reach_rmax_max=int(config.get('batch_renorm','reach_rmax_max'))
            self.start_relaxing_dmax=int(config.get('batch_renorm','start_relaxing_dmax'))
            self.reach_dmax_max=int(config.get('batch_renorm','reach_dmax_max'))       
        except:
            sys.exit("Check configuration file config.txt. At least one required option is missing in section [batch_renorm].")        
        #-----finish reading from config.txt------------------------
        
        self.train_step_num=tf.Variable(0.0, dtype=tf.dtypes.float32)
        self.rmax=tf.Variable(1.0, dtype=tf.dtypes.float32)
        self.dmax=tf.Variable(0.0, dtype=tf.dtypes.float32)
                        
        self.start_conv=start_conv
        self.start_pool=start_pool
        self.structure_of_blocks=structure_of_blocks
        self.lengths_of_blocks=lengths_of_blocks
                
        
        
        #_start denotes a few layers coming before ResNet blocks
        
        start_conv_filter_size=self.start_conv[0]
        start_conv_depth=self.start_conv[1]
        
        self.w_start=tf.Variable(tf.random.normal(shape=[start_conv_filter_size,start_conv_filter_size,3,start_conv_depth],stddev=tf.math.sqrt(tf.cast(2/(start_conv_filter_size*start_conv_filter_size*3),dtype=tf.dtypes.float32))))
        self.b_start=tf.Variable(tf.zeros(shape=[1,1,1,start_conv_depth]))
        self.beta_start=tf.Variable(tf.zeros(shape=[1,1,1,start_conv_depth]))
        self.gamma_start=tf.Variable(tf.ones(shape=[1,1,1,start_conv_depth]))
        self.mu_start=tf.Variable(tf.zeros(shape=[1,1,1,start_conv_depth]))
        self.sigma_start=tf.Variable(tf.ones(shape=[1,1,1,start_conv_depth]))
        
        self.w=[]
        self.b=[]
        self.beta=[]
        self.gamma=[]
        self.mu=[]
        self.sigma=[]
        
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
                    self.sigma.append(tf.Variable(tf.ones(shape=[1,1,1,depth])))
                    
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
            mu_batch=tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[0]
            sigma_batch=tf.math.sqrt(var_epsilon+tf.nn.moments(Z_start,axes=[0,1,2],keepdims=True)[1])
            
            mu_to_apply, V_to_apply = self.get_mu_and_V(self.sigma_start,self.mu_start,sigma_batch,mu_batch)
            
            M_start=tf.nn.batch_normalization(Z_start,mean=mu_to_apply,variance=V_to_apply,offset=self.beta_start,scale=self.gamma_start,variance_epsilon=var_epsilon)
            
            self.mu_start=self.moving_avgs_momentum*self.mu_start+(1-self.moving_avgs_momentum)*mu_batch
            self.sigma_start=self.moving_avgs_momentum*self.sigma_start+(1-self.moving_avgs_momentum)*sigma_batch
            
        elif(mode=='inference'):
            #M_start from Z_start, mean=mu, variance=V
            M_start=tf.nn.batch_normalization(Z_start,mean=self.mu_start,variance=self.sigma_start**2,offset=self.beta_start,scale=self.gamma_start,variance_epsilon=var_epsilon)
                
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
                        
                        mu_batch=tf.nn.moments(L[-1],axes=[0,1,2],keepdims=True)[0]
                        sigma_batch=tf.math.sqrt(var_epsilon+tf.nn.moments(L[-1], axes=[0,1,2], keepdims=True)[1])
                        
                        mu_to_apply, V_to_apply = self.get_mu_and_V(self.sigma[num],self.mu[num],sigma_batch,mu_batch)
                        
                        L.append(tf.nn.batch_normalization(L[-1],mean=mu_to_apply, variance=V_to_apply, offset=self.beta[num], scale=self.gamma[num], variance_epsilon=var_epsilon))
                        
                        self.mu[num]=self.moving_avgs_momentum*self.mu[num]+(1-self.moving_avgs_momentum)*mu_batch
                        self.sigma[num]=self.moving_avgs_momentum*self.sigma[num]+(1-self.moving_avgs_momentum)*sigma_batch
                        
                        
                    elif(mode=='inference'):
                        #L from L[-1], mean=mu, variance=V
                        L.append(tf.nn.batch_normalization(L[-1],mean=self.mu[num],variance=self.sigma[num]**2,offset=self.beta[num],scale=self.gamma[num],variance_epsilon=var_epsilon))
   
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
        
        beta=0.00001
                        
        N=len(X)
        A=self.__call__(X)
        YTA=tf.matmul(tf.transpose(Y),tf.math.log((1-beta)*A+beta))
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
        
        self.train_step_num.assign_add(1.0)
        
        if(self.train_step_num<self.start_relaxing_rmax):
            rmax=1.0
        elif(self.train_step_num>self.reach_rmax_max):
            rmax=self.rmax_max
        else:
            rmax=1.0+((self.train_step_num-self.start_relaxing_rmax)/(self.reach_rmax_max-self.start_relaxing_rmax))*(self.rmax_max-1.0)
            
        if(self.train_step_num<self.start_relaxing_dmax):
            dmax=0.0
        elif(self.train_step_num>self.reach_dmax_max):
            dmax=self.dmax_max
        else:
            dmax=((self.train_step_num-self.start_relaxing_dmax)/(self.reach_dmax_max-self.start_relaxing_dmax))*self.dmax_max
            
        self.rmax.assign(rmax)
        self.dmax.assign(dmax)
        
    @tf.function
    def get_mu_and_V(self,sigma_moving,mu_moving,sigma_batch,mu_batch):
        
        r=tf.stop_gradient(tf.clip_by_value(sigma_batch/sigma_moving, 1/self.rmax, self.rmax))
        d=tf.stop_gradient(tf.clip_by_value((mu_batch-mu_moving)/sigma_moving, -self.dmax, self.dmax))
        
        V=(sigma_batch/r)**2
        mu=mu_batch-d*sigma_batch/r
        
        return mu, V

        
        
        
