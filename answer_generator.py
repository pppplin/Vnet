import tensorflow as tf

import image_processor
import text_processor
from tensorflow.contrib import layers

class Answer_Generator():
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.img_dim = params['img_dim']
        self.hidden_dim = params['hidden_dim']
        self.rnn_size = params['rnn_size']
        self.rnn_layer = params['rnn_layer']
        self.init_bound = params['init_bound']
        self.num_output = params['num_output']
        self.dropout_rate = params['dropout_rate']
        self.ans_vocab_size = params['ans_vocab_size']
        self.max_que_length = params['max_que_length']
        self.img_feature_size = params['img_feature_size']
        self.attention_round = params['attention_round']
        self.attention_hidden_dim = params['attention_hidden_dim']
        
        self.que_vocab_size = params['que_vocab_size']
        self.que_embed_size = params['que_embed_size']
        self.img_processor = image_processor.Vgg16()
        self.que_processor = text_processor.Deeper_LSTM({
            'rnn_size': self.rnn_size,
            'rnn_layer': self.rnn_layer,
            'init_bound': self.init_bound,
            'que_vocab_size': params['que_vocab_size'],
            'que_embed_size': params['que_embed_size'],
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'max_que_length': self.max_que_length
        })

        self.que_W = tf.Variable(tf.random_uniform([2 * self.rnn_layer * self.rnn_size, self.hidden_dim], \
                                                   -self.init_bound, self.init_bound),
                                 name='text_W')
        self.que_b = tf.Variable(tf.random_uniform([self.hidden_dim], -self.init_bound, self.init_bound), name='text_b')
        self.img_W = tf.Variable(tf.random_uniform([self.img_dim, self.hidden_dim], \
                                                   -self.init_bound, self.init_bound),
                                 name='img_W')
        self.img_b = tf.Variable(tf.random_uniform([self.hidden_dim], -self.init_bound, self.init_bound), name='img_b')
        self.score_W = tf.Variable(tf.random_uniform([self.hidden_dim, self.num_output], \
                                                     -self.init_bound, self.init_bound),
                                   name='score_W')
        
        self.score_b = tf.Variable(tf.random_uniform([self.num_output], -self.init_bound, self.init_bound, name='score_b'))

        self.score_b_h = tf.Variable(
            tf.random_uniform([self.hidden_dim], -self.init_bound, self.init_bound, name='score_bh'))

        self.score_W_uv = tf.Variable(
            tf.random_uniform([self.rnn_size, self.hidden_dim], -self.init_bound, self.init_bound, \
                              name='score_W_uv'))
        self.score_W_uq = tf.Variable(
            tf.random_uniform([self.rnn_size, self.hidden_dim], -self.init_bound, self.init_bound, \
                              name='score_W_uq'))
        
        self.build_attention_model()


    def train_model(self):
        # The baseline.(didn't change, not using) 
        img_state = tf.placeholder('float32', [None, self.img_dim], name='img_state')
        label_batch = tf.placeholder('float32', [None, self.ans_vocab_size], name='label_batch')
        real_size = tf.shape(img_state)[0]

        que_state, sentence_batch = self.que_processor.train()
        drop_que_state = tf.nn.dropout(que_state, 1 - self.dropout_rate)
        drop_que_state = tf.reshape(drop_que_state, [self.batch_size, 2 * self.rnn_layer * self.rnn_size])
        linear_que_state = tf.nn.xw_plus_b(drop_que_state, self.que_W, self.que_b)
        que_feature = tf.tanh(linear_que_state)

        drop_img_state = tf.nn.dropout(img_state, 1 - self.dropout_rate)
        linear_img_state = tf.nn.xw_plus_b(drop_img_state, self.img_W, self.img_b)
        img_feature = tf.tanh(linear_img_state)

        score = tf.mul(que_feature, img_feature)
        drop_score = tf.nn.dropout(score, 1 - self.dropout_rate)
        logits = tf.nn.xw_plus_b(drop_score, self.score_W, self.score_b)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, label_batch, name='entropy')
        ans_probability = tf.nn.softmax(logits, name='answer_prob')

        predict = tf.argmax(ans_probability, 1)
        correct_predict = tf.equal(tf.argmax(ans_probability, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        loss = tf.reduce_sum(cross_entropy, name='loss')
        #for test
        return loss, accuracy, predict, img_state, sentence_batch, label_batch
    
    def build_attention_model(self):
        self.W_uv = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
                                             self.init_bound, name='W_uv'))
        self.W_uq = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
                                             self.init_bound, name='W_uq'))
        self.b_u = tf.Variable(tf.random_uniform([1], -self.init_bound, \
                                            self.init_bound), name='b_u')
        self.W_c = tf.Variable(tf.random_uniform([self.rnn_size, self.rnn_size], -self.init_bound, \
                                            self.init_bound, name='W_c'))
        self.W_hv = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], \
                                             -self.init_bound,self.init_bound), name='W_hv')
        #(N,k)
        #Bias for H_v in paper
        self.b_hv = tf.Variable(tf.random_uniform([self.img_feature_size, self.attention_hidden_dim], -self.init_bound, \
                                             self.init_bound), name='b_hv')
        #(d,k)
        #W_q in paper
        self.W_hq = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], -self.init_bound, \
                                             self.init_bound), name='W_hq')
        #(T,k)
        #Bias for H_q in paper
        self.b_hq = tf.Variable(tf.random_uniform([self.max_que_length, self.attention_hidden_dim], -self.init_bound, \
                                             self.init_bound), name='b_hq')
        
        #w_hv in paper
        self.W_av = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
                                             self.init_bound), name='W_av')
        
        #w_hq in paper
        self.W_aq = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
                                             self.init_bound), name='W_aq')
        #(N,1)
        #bias for a_v in paper
        self.b_av = tf.Variable(tf.random_uniform([self.img_feature_size, 1], -self.init_bound, \
                                             self.init_bound), name='b_av')
        #(T,1)
        #bias for a_q in paper
        self.b_aq = tf.Variable(tf.random_uniform([self.max_que_length, 1], -self.init_bound, \
                                             self.init_bound), name='b_aq')   
        
        

    def train_attention_model(self):
        #ALWAYS USING THIS ONE
        #v/V for image, q/Q for question
        #KEY COMPONENTS: v, q, V, Q, C
        #v, q is the attentioned visual/question vector.
        #C is the affinity map in the paper
        #V, Q is the attentioned vector before reduce sum. feed as input for next attention procedure.
        V0 = tf.placeholder('float32', [None, self.img_feature_size, self.rnn_size], name='img_state')
        # # feed_V0 = layers.batch_norm(V0)
        # drop_V0 = tf.nn.dropout(V0, 1 - self.dropout_rate)#nan and inf...
        # norm_V0 = layers.batch_norm(drop_V0)
        # feed_V0 = tf.tanh(norm_V0)#NAN
        
        label_batch = tf.placeholder('float32', [None, self.ans_vocab_size], name='label_batch')
        Q0, sentence_batch = self.que_processor.train()
        
        feed_V0 = V0
        with tf.variable_scope("attention"):
            v, q, V, Q, C_update= self.attention(feed_V0, Q0)
            C_old = None
            for i in range(self.attention_round):
                #self.attention_round=0 to get the model in paper
                C_old = self.memory_cell(v, q, C_update=C_update, C_old=C_old)
                v, q, V, Q, C_update= self.attention(V, Q, C=C_old)
        
        #following is regular process to get prediction        
        score = tf.tanh(tf.matmul(v, self.score_W_uv) + tf.matmul(q, self.score_W_uq) + self.score_b_h)
        logits = tf.nn.xw_plus_b(score, self.score_W, self.score_b)
        logits = layers.batch_norm(logits)
        #not NAN?
        logits = tf.tanh(logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, label_batch, name='entropy')
        
        ans_probability = tf.nn.softmax(logits, name='answer_prob')
        predict = tf.argmax(ans_probability, 1)
        correct_predict = tf.equal(tf.argmax(ans_probability, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        loss = tf.reduce_sum(cross_entropy, name='loss')

        return loss, accuracy, predict, V0, sentence_batch, label_batch
        # return loss, accuracy, predict, V0, sentence_batch, label_batch, feed_V0

#     def memory_cell(self, v, q, C_update=None, C_old=None):
#         #USE TO UPDATE THE AFFINITY MAP
#         #Accept v,q
#         if C_old==None:
#             return C_update
#         W_uv = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
#                                              self.init_bound, name='W_uv'))
#         W_uq = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
#                                              self.init_bound, name='W_uq'))
#         b_u = tf.Variable(tf.random_uniform([1], -self.init_bound, \
#                                             self.init_bound), name='b_u')
#         #(bs,1)
#         #the forget gate u = sigmind(W_uv*v+W_uq*q+b_u). to determine how mush to update in the affinity map
#         u = tf.sigmoid(tf.reduce_sum(tf.matmul(v, W_uv) + tf.matmul(q, W_uq) + b_u))
#         C = tf.scalar_mul(1 - u, C_old) + tf.scalar_mul(u, C_update)
#         return C
    
#     def attention(self, V, Q , C=None):
#         # AN IMPLEMENTATION OF: Hierarchical Question-Image Co-Attention(Parallel)
#         #tf.einsum('ijk,lk->ijl',M1,M2): JUST a way to represent matirx multiplication with more flexible shapes. 
#         #C is the affinity map
#         W_c = tf.Variable(tf.random_uniform([self.rnn_size, self.rnn_size], -self.init_bound, \
#                                             self.init_bound, name='W_c'))
#         update = True
#         #compute C
#         if C == None:
#             update= False
#             # (N, T)
#             #alter 
#             # C = tf.batch_matmul(tf.einsum('ijk,lk->ijl',V,W_c), Q, adj_y=True)
#             # C = tf.reduce_sum(C,0)
#             C = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl',V,W_c), Q)
        
#         #FOLLOWING: implementation of (4) and (5) in the paper.    
#         #(k,d)
#         #W_v in paper
#         W_hv = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], \
#                                              -self.init_bound,self.init_bound), name='W_hv')
#         #(N,k)
#         #Bias for H_v in paper
#         b_hv = tf.Variable(tf.random_uniform([self.img_feature_size, self.attention_hidden_dim], -self.init_bound, \
#                                              self.init_bound), name='b_hv')
#         #(d,k)
#         #W_q in paper
#         W_hq = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], -self.init_bound, \
#                                              self.init_bound), name='W_hq')
#         #(T,k)
#         #Bias for H_q in paper
#         b_hq = tf.Variable(tf.random_uniform([self.max_que_length, self.attention_hidden_dim], -self.init_bound, \
#                                              self.init_bound), name='b_hq')

#         #(N, k(attention_hidden_dim))
#         #H_v in paper
#         H_v = tf.nn.relu(tf.einsum('aij,kj->ik', V, W_hv) + tf.einsum('ij,ki->jk', tf.einsum('aij,ki->kj', Q, C), W_hq) + b_hv)
        
#         #(T, k(attention_hidden_dim))
#         #H_q in paper
#         H_q = tf.nn.relu(tf.einsum('aij,kj->ik', Q, W_hq)+ tf.einsum('ij,ki->jk', tf.einsum('aij,ik->kj', V, C), W_hv)+ b_hq)
        
#         #w_hv in paper
#         W_av = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
#                                              self.init_bound), name='W_av')
        
#         #w_hq in paper
#         W_aq = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
#                                              self.init_bound), name='W_aq')
#         #(N,1)
#         #bias for a_v in paper
#         b_av = tf.Variable(tf.random_uniform([self.img_feature_size, 1], -self.init_bound, \
#                                              self.init_bound), name='b_av')
#         #(T,1)
#         #bias for a_q in paper
#         b_aq = tf.Variable(tf.random_uniform([self.max_que_length, 1], -self.init_bound, \
#                                              self.init_bound), name='b_aq')
#         #(N,1)
#         #a_v in paper
#         a_v = tf.nn.softmax(tf.matmul(H_v, W_av) + b_av)
#         #(T,1)
#         #a_q in paper
#         a_q = tf.nn.softmax(tf.matmul(H_q, W_aq) + b_aq)
        
#         #v jian in paper
#         v = tf.einsum('ijk,j->ik', V, tf.squeeze(a_v))
#         #q jian in paper
#         q = tf.einsum('ijk,j->ik', Q, tf.squeeze(a_q))
        
#         #v jian before reduce sum 
#         V_new = tf.mul(V, a_v)   #(bs, N, d)
#         #q jian before reduce sum
#         Q_new = tf.mul(Q, a_q)  #(bs, T, d)
#         #(bs, d)

#         #maybe rescale?????? then keep attention before sigmoid
#         # V = tf.mul(V, v)#(bs, N, d)
#         # Q = tf.mul(V, q)#(bs, N, d)

#         #Update C
#         if update:
#             C = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl',V,W_c), Q)

#         return (v, q, V_new, Q_new, C)

    # def memory_cell(self, v, q, C_update=None, C_old=None):
    #     #USE TO UPDATE THE AFFINITY MAP
    #     #Accept v,q
    #     if C_old==None:
    #         return C_update
    #     W_uv = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
    #                                          self.init_bound, name='W_uv'))
    #     W_uq = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
    #                                          self.init_bound, name='W_uq'))
    #     b_u = tf.Variable(tf.random_uniform([1], -self.init_bound, \
    #                                         self.init_bound), name='b_u')
    #     #(bs,1)
    #     #the forget gate u = sigmind(W_uv*v+W_uq*q+b_u). to determine how mush to update in the affinity map
    #     u = tf.sigmoid(tf.reduce_sum(tf.matmul(v, W_uv) + tf.matmul(q, W_uq) + b_u))
    #     C = tf.scalar_mul(1 - u, C_old) + tf.scalar_mul(u, C_update)
    #     return C
def memory_cell(self, v, q, C_update=None, C_old=None):
        #USE TO UPDATE THE AFFINITY MAP
        #Accept v,q
        if C_old==None:
            return C_update
        W_uv = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
                                             self.init_bound, name='W_uv'))
        W_uq = tf.Variable(tf.random_uniform([self.rnn_size,1], -self.init_bound, \
                                             self.init_bound, name='W_uq'))
        b_u = tf.Variable(tf.random_uniform([1], -self.init_bound, \
                                            self.init_bound), name='b_u')
        #(bs,1)
        #the forget gate u = sigmind(W_uv*v+W_uq*q+b_u). to determine how mush to update in the affinity map
        u = tf.sigmoid(tf.reduce_sum(tf.matmul(v, W_uv) + tf.matmul(q, W_uq) + b_u))
        C = tf.scalar_mul(1 - u, C_old) + tf.scalar_mul(u, C_update)
        return C
    
    def attention(self, V, Q , C=None):
        # AN IMPLEMENTATION OF: Hierarchical Question-Image Co-Attention(Parallel)
        #tf.einsum('ijk,lk->ijl',M1,M2): JUST a way to represent matirx multiplication with more flexible shapes. 
        #C is the affinity map
        W_c = tf.Variable(tf.random_uniform([self.rnn_size, self.rnn_size], -self.init_bound, \
                                            self.init_bound, name='W_c'))
        update = True
        #compute C
        if C == None:
            update= False
            # (N, T)
            #alter 
            # C = tf.batch_matmul(tf.einsum('ijk,lk->ijl',V,W_c), Q, adj_y=True)
            # C = tf.reduce_sum(C,0)
            C = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl',V,W_c), Q)
        
        #FOLLOWING: implementation of (4) and (5) in the paper.    
        #(k,d)
        #W_v in paper
        W_hv = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], \
                                             -self.init_bound,self.init_bound), name='W_hv')
        #(N,k)
        #Bias for H_v in paper
        b_hv = tf.Variable(tf.random_uniform([self.img_feature_size, self.attention_hidden_dim], -self.init_bound, \
                                             self.init_bound), name='b_hv')
        #(d,k)
        #W_q in paper
        W_hq = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], -self.init_bound, \
                                             self.init_bound), name='W_hq')
        #(T,k)
        #Bias for H_q in paper
        b_hq = tf.Variable(tf.random_uniform([self.max_que_length, self.attention_hidden_dim], -self.init_bound, \
                                             self.init_bound), name='b_hq')

        #(N, k(attention_hidden_dim))
        #H_v in paper
        H_v = tf.nn.relu(tf.einsum('aij,kj->ik', V, W_hv) + tf.einsum('ij,ki->jk', tf.einsum('aij,ki->kj', Q, C), W_hq) + b_hv)
        
        #(T, k(attention_hidden_dim))
        #H_q in paper
        H_q = tf.nn.relu(tf.einsum('aij,kj->ik', Q, W_hq)+ tf.einsum('ij,ki->jk', tf.einsum('aij,ik->kj', V, C), W_hv)+ b_hq)
        
        #w_hv in paper
        W_av = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
                                             self.init_bound), name='W_av')
        
        #w_hq in paper
        W_aq = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
                                             self.init_bound), name='W_aq')
        #(N,1)
        #bias for a_v in paper
        b_av = tf.Variable(tf.random_uniform([self.img_feature_size, 1], -self.init_bound, \
                                             self.init_bound), name='b_av')
        #(T,1)
        #bias for a_q in paper
        b_aq = tf.Variable(tf.random_uniform([self.max_que_length, 1], -self.init_bound, \
                                             self.init_bound), name='b_aq')
        #(N,1)
        #a_v in paper
        a_v = tf.nn.softmax(tf.matmul(H_v, W_av) + b_av)
        #(T,1)
        #a_q in paper
        a_q = tf.nn.softmax(tf.matmul(H_q, W_aq) + b_aq)
        
        #v jian in paper
        v = tf.einsum('ijk,j->ik', V, tf.squeeze(a_v))
        #q jian in paper
        q = tf.einsum('ijk,j->ik', Q, tf.squeeze(a_q))
        
        #v jian before reduce sum 
        V_new = tf.mul(V, a_v)   #(bs, N, d)
        #q jian before reduce sum
        Q_new = tf.mul(Q, a_q)  #(bs, T, d)
        #(bs, d)

        #maybe rescale?????? then keep attention before sigmoid
        # V = tf.mul(V, v)#(bs, N, d)
        # Q = tf.mul(V, q)#(bs, N, d)

        #Update C
        if update:
            C = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl',V,W_c), Q)

        return (v, q, V_new, Q_new, C)
    
    def attention(self, V, Q , C=None):
        # AN IMPLEMENTATION OF: Hierarchical Question-Image Co-Attention(Parallel)
        #tf.einsum('ijk,lk->ijl',M1,M2): JUST a way to represent matirx multiplication with more flexible shapes. 
        #C is the affinity map
        W_c = tf.Variable(tf.random_uniform([self.rnn_size, self.rnn_size], -self.init_bound, \
                                            self.init_bound, name='W_c'))
        update = True
        #compute C
        if C == None:
            update= False
            # (N, T)
            #alter 
            # C = tf.batch_matmul(tf.einsum('ijk,lk->ijl',V,W_c), Q, adj_y=True)
            # C = tf.reduce_sum(C,0)
            C = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl',V,W_c), Q)
        
        #FOLLOWING: implementation of (4) and (5) in the paper.    
        #(k,d)
        #W_v in paper
        W_hv = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], \
                                             -self.init_bound,self.init_bound), name='W_hv')
        #(N,k)
        #Bias for H_v in paper
        b_hv = tf.Variable(tf.random_uniform([self.img_feature_size, self.attention_hidden_dim], -self.init_bound, \
                                             self.init_bound), name='b_hv')
        #(d,k)
        #W_q in paper
        W_hq = tf.Variable(tf.random_uniform([self.attention_hidden_dim, self.rnn_size], -self.init_bound, \
                                             self.init_bound), name='W_hq')
        #(T,k)
        #Bias for H_q in paper
        b_hq = tf.Variable(tf.random_uniform([self.max_que_length, self.attention_hidden_dim], -self.init_bound, \
                                             self.init_bound), name='b_hq')

        #(N, k(attention_hidden_dim))
        #H_v in paper
        H_v = tf.nn.relu(tf.einsum('aij,kj->ik', V, W_hv) + tf.einsum('ij,ki->jk', tf.einsum('aij,ki->kj', Q, C), W_hq) + b_hv)
        
        #(T, k(attention_hidden_dim))
        #H_q in paper
        H_q = tf.nn.relu(tf.einsum('aij,kj->ik', Q, W_hq)+ tf.einsum('ij,ki->jk', tf.einsum('aij,ik->kj', V, C), W_hv)+ b_hq)
        
        #w_hv in paper
        W_av = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
                                             self.init_bound), name='W_av')
        
        #w_hq in paper
        W_aq = tf.Variable(tf.random_uniform([self.attention_hidden_dim,1], -self.init_bound, \
                                             self.init_bound), name='W_aq')
        #(N,1)
        #bias for a_v in paper
        b_av = tf.Variable(tf.random_uniform([self.img_feature_size, 1], -self.init_bound, \
                                             self.init_bound), name='b_av')
        #(T,1)
        #bias for a_q in paper
        b_aq = tf.Variable(tf.random_uniform([self.max_que_length, 1], -self.init_bound, \
                                             self.init_bound), name='b_aq')
        #(N,1)
        #a_v in paper
        a_v = tf.nn.softmax(tf.matmul(H_v, W_av) + b_av)
        #(T,1)
        #a_q in paper
        a_q = tf.nn.softmax(tf.matmul(H_q, W_aq) + b_aq)
        
        #v jian in paper
        v = tf.einsum('ijk,j->ik', V, tf.squeeze(a_v))
        #q jian in paper
        q = tf.einsum('ijk,j->ik', Q, tf.squeeze(a_q))
        
        #v jian before reduce sum 
        V_new = tf.mul(V, a_v)   #(bs, N, d)
        #q jian before reduce sum
        Q_new = tf.mul(Q, a_q)  #(bs, T, d)
        #(bs, d)

        #maybe rescale?????? then keep attention before sigmoid
        # V = tf.mul(V, v)#(bs, N, d)
        # Q = tf.mul(V, q)#(bs, N, d)

        #Update C
        if update:
            C = tf.einsum('aik,ajk->ij', tf.einsum('ijk,lk->ijl',V,W_c), Q)

        return (v, q, V_new, Q_new, C)

