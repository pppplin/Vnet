import tensorflow as tf

import image_processor
import text_processor

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


        self.que_W = tf.Variable(tf.random_uniform([2 * self.rnn_layer * self.rnn_size , self.hidden_dim], \
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

    def train_model(self):
        img_state = tf.placeholder('float32', [None, self.img_dim], name='img_state')
        label_batch = tf.placeholder('float32', [None, self.ans_vocab_size], name='label_batch')
        real_size = tf.shape(img_state)[0]

        que_state, sentence_batch = self.que_processor.train()
        drop_que_state = tf.nn.dropout(que_state, 1 - self.dropout_rate)
        drop_que_state = tf.reshape(drop_que_state, [real_size, 2 * self.rnn_layer * self.rnn_size])
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
        
        return loss, accuracy, predict, img_state, sentence_batch, label_batch
    
    def train_attention_model(self):
        img_state = tf.placeholder('float32', [None, self.img_feature_size, self.img_dim], name='img_state')
        label_batch = tf.placeholder('float32', [None, self.ans_vocab_size], name='label_batch')
        real_size = tf.shape(img_state)[0]

        que_state, sentence_batch = self.que_processor.train()
        drop_que_state = tf.nn.dropout(que_state, 1 - self.dropout_rate)
        drop_que_state = tf.reshape(drop_que_state, [real_size, 2 * self.rnn_layer * self.rnn_size])
        linear_que_state = tf.nn.xw_plus_b(drop_que_state, self.que_W, self.que_b)
        Q0 = tf.tanh(linear_que_state)

        drop_img_state = tf.nn.dropout(img_state, 1 - self.dropout_rate)
        linear_img_state = tf.nn.xw_plus_b(drop_img_state, self.img_W, self.img_b)
        V0 = tf.tanh(linear_img_state)
        
        with tf.variable_scope("attention"):
            v, q, V, Q, C = attention(V0, Q0, C=None)#if C is none then return else dont return
            for i in range(self.attention_round):  
                C_new = memory_cell(v, q, V, Q, C)#update C
                v, q, V, Q = attention(V, Q, C=C_new)# update v,q, V,Q
                C = C_new
    
        
        score = tf.mul(que_feature, img_feature)#score = tf.tanh(W_h(v+q))...revise the rest
        drop_score = tf.nn.dropout(score, 1 - self.dropout_rate)
        logits = tf.nn.xw_plus_b(drop_score, self.score_W, self.score_b)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, label_batch, name='entropy')
        ans_probability = tf.nn.softmax(logits, name='answer_prob')

        predict = tf.argmax(ans_probability, 1)
        correct_predict = tf.equal(tf.argmax(ans_probability, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        loss = tf.reduce_sum(cross_entropy, name='loss')
        
        return loss, accuracy, predict, img_state, sentence_batch, label_batch

    
    def attention(self, v, q, V, Q, C=None):
        #C is the affinity map
        bs = V.get_shape()[0]
        W_c = tf.Variable(tf.random_uniform([self.rnn_size, self.rnn_size], -self.init_bound, \
                                            self.init_bound, name='W_c'))
        # Q = tf.transpose(Q, perm=[0, 2, 1])
        W_c_batch = tf.pack([W_c]*bs, axis=0)
        if C == None:
            return_c = True
            #compute C
            Q = tf.transpose(Q, perm=[0, 2, 1])
            # (bs, N, T)
            C = tf.batch_matmul(tf.batch_matmul(V, W_c_batch) , Q) 
            #and return C, transform Q back
            Q = tf.transpose(Q, perm=[0, 2, 1])
        #NO dropout
        #(d,k)
        W_hv = tf.Variable(tf.randim_uniform([self.rnn_size, self.hidden_dim]), -self.init_bound,\
                          self.init_bound, name = 'W_hv')
        #(N,k)
        b_hv = tf.Variable(tf.randim_uniform([self.img_feature_size, self.hidden_dim]), -self.init_bound,\
                          self.init_bound, name = 'b_hv')
        #(d,k)
        W_hq = tf.Variable(tf.randim_uniform([self.rnn_size, self.hidden_dim]), -self.init_bound,\
                          self.init_bound, name = 'W_hq')
        #(T,k)
        b_hq = tf.Variable(tf.randim_uniform([self.max_que_length, self.hidden_dim]), -self.init_bound,\
                          self.init_bound, name = 'b_hq')
        
        W_hv_batch = tf.pack([W_hv]*bs)
        # (bs, d,k)
        W_hq_batch = tf.pack([W_hq]*bs)
        #(bs, N, k(hidden_dim))
        H_v = tf.tanh(tf.batch_matmul(V,W_hv_batch)+ tf.batch_matmul(tf.batch_matmul(C,Q),W_hq_batch)+b_hv)
        #(bs, T, k(hidden_dim)) 
        H_q = tf.tanh(tf.batch_matmul(Q,W_hq_batch)+ tf.batch_matmul(tf.batch_matmul(tf.transpose(C, perm=[0,2,1]),V),\
                                                                     W_hv_batch)+b_hq)
        
        W_av = tf.Variable(tf.randim_uniform([self.hidden_dim, 1]), -self.init_bound,\
                          self.init_bound, name = 'W_av')
        W_aq = tf.Variable(tf.randim_uniform([self.hidden_dim, 1]), -self.init_bound,\
                          self.init_bound, name = 'W_aq')
        #(N,1)
        b_av = tf.Variable(tf.randim_uniform([self.img_feature_size, 1]), -self.init_bound,\
                          self.init_bound, name = 'b_av')
        #(T,1)
        b_aq = tf.Variable(tf.randim_uniform([self.max_que_length, 1]), -self.init_bound,\
                          self.init_bound, name = 'b_aq')
        #(bs,N,1)
        a_v = tf.sigmoid(tf.batch_matmul(H_v,tf.pack([W_av]*bs))+b_av)
        #(bs,T,1)
        a_q = tf.sigmoid(tf.batch_matmul(H_q,tf.pack([W_aq]*bs))+b_aq)
        
        V = tf.mul(V, a_v)#(bs, N, d)
        Q = tf.mul(Q, a_q)#(bs, T, d)
        #(bs, d)
        v = tf.squeeze(tf.reduce_sum(V, 1))
        q = tf.squeeze(tf.reduce_sum(Q, 1))
        #maybe rescale?????? then keep attention before sigmoid
        # V = tf.mul(V, v)#(bs, N, d)
        # Q = tf.mul(V, q)#(bs, N, d)
        
        if return_c = True:
            return (v, q, V, Q, C)
        else:
            return (v, q, V, Q)
            
        