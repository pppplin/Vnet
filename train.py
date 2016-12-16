from gpu import define_gpu
define_gpu(1)

import tensorflow as tf
import numpy as np
import data_loader
import argparse
import os

import image_processor
import answer_generator

def build_batch(batch_num, batch_size, img_feature, img_id_map, qa_data, vocab_data, split, embedd_size):
    qa = qa_data[split]
    batch_start = (batch_num * batch_size) % len(qa)
    batch_end = min(len(qa), batch_start + batch_size)
    size = batch_end - batch_start
    sentence = np.ndarray((size, vocab_data['max_que_length']), dtype='int32')
    answer = np.zeros((size, len(vocab_data['ans_vocab'])))
    img = np.ndarray((size, 7,7, embedd_size))
    # img = np.ndarray((size, 4096))
    counter = 0
    for i in range(batch_start, batch_end):
        sentence[counter, :] = qa[i]['question'][:]
        answer[counter, qa[i]['answer']] = 1.0
        img_index = img_id_map[qa[i]['image_id']]
        img[counter, :] = img_feature[img_index][:]
        counter += 1
    return sentence, answer, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/zs/NLP/Vnet/data/', help='Data Directory')
    parser.add_argument('--log_dir', type=str, default='log', help='Checkpoint File Directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')

    parser.add_argument('--batch_size', type=int, default=32, help='Image Training Batch Size,64')
    parser.add_argument('--num_output', type=int, default=1000, help='Number of Output')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout Rate')
    parser.add_argument('--init_bound', type=float, default=0.5, help='Parameter Initialization Distribution Bound, 0.5')

    parser.add_argument('--hidden_dim', type=int, default=128, help='RNN Hidden State Dimension, 1024')
    #TODO: change to embed size!!!!!!!!
    parser.add_argument('--rnn_size', type=int, default=512, help='Size of RNN Cell(C and h), question embedding size,512')
    parser.add_argument('--rnn_layer', type=int, default=1, help='Number of RNN Layers,2')
    parser.add_argument('--que_embed_size', type=int, default=20, help='Question Embedding Dimension, 200')

    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning Rate, 1e-4')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning Rate Decay Factor')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of Training Epochs')
    parser.add_argument('--grad_norm', type=int, default=5, help='Maximum Norm of the Gradient')
    
    parser.add_argument('--img_feature_size', type=int, default=49, help='7*7, wide*height img feature after deconv')
    parser.add_argument('--attention_round', type=int, default=0, help='number of attention round')
    parser.add_argument('--attention_hidden_dim', type=int, default=16, help='k in paper, attention hidden dim')
    #MUST BE TRUE: preprocess only fit attention model
    parser.add_argument('--use_attention', type=bool, default=True, help='whether to use attention model')
    
    parser.add_argument('--my_img_data_dir', type=str, default='/home/weilin/VQA/Vnet/data/', help='Data Directory')
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.makedirs(os.path.join(args.log_dir, 'model'))
        os.makedirs(os.path.join(args.log_dir, 'summary'))

    print 'Reading Question Answer Data'
    qa_data, vocab_data = data_loader.load_qa_data(args.data_dir, args.top_num)

    train_img_feature, train_img_id_list = image_processor.VGG_16_extract('train', args)
    print(train_img_feature.shape)
    print 'Building Image ID Map and Answer Map'
    img_id_map = {}
    
    for i in xrange(len(train_img_id_list)):
        img_id_map[train_img_id_list[i]] = i
    ans_map = {vocab_data['ans_vocab'][ans] : ans for ans in vocab_data['ans_vocab']}
    
    print 'Building Answer Generator Model'
    generator = answer_generator.Answer_Generator({
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'img_dim': train_img_feature.shape[1],
        'rnn_size': args.rnn_size,
        'rnn_layer': args.rnn_layer,
        'que_embed_size': args.que_embed_size,
        'que_vocab_size': len(vocab_data['que_vocab']),
        'ans_vocab_size': len(vocab_data['ans_vocab']),
        'max_que_length': vocab_data['max_que_length'],
        'num_output': args.num_output,
        'dropout_rate': args.dropout_rate,
        'data_dir': args.data_dir,
        'top_num': args.top_num,
        'init_bound': args.init_bound,
        'img_feature_size': args.img_feature_size,
        'attention_round': args.attention_round,
        'attention_hidden_dim': args.attention_hidden_dim
        })

    lr = args.learning_rate
    if args.use_attention:
        loss, accuracy, predict, feed_img, feed_que, feed_label= generator.train_attention_model()        
        # loss, accuracy, predict, feed_img, feed_que, feed_label, feed_V0= generator.train_attention_model()        
    else:
        loss, accuracy, predict, feed_img, feed_que, feed_label = generator.train_model()

    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()

    tf.initialize_all_variables().run(session=sess)

    train_summary_writer = tf.train.SummaryWriter(os.path.join(args.log_dir, "summaries", "train"), sess.graph)
    val_summary_writer = tf.train.SummaryWriter(os.path.join(args.log_dir, "summaries", "val"), sess.graph)

    print 'Training Start...'
    saver = tf.train.Saver()
    for epoch in range(args.num_epoch):
        print 'Epoch %d #############' % epoch
        train_batch_num = 0
        val_batch_num = 0
        val_acc_list = []
        while train_batch_num * args.batch_size < len(qa_data['train']):
            que_batch, ans_batch, img_batch = build_batch(train_batch_num, args.batch_size, \
                                                train_img_feature, img_id_map, qa_data, vocab_data, 'train', args.rnn_size)
            
            _, loss_value, acc, pred = sess.run([train_op, loss, accuracy, predict],
                                                    feed_dict={
                                                        feed_img: img_batch,
                                                        feed_que: que_batch,
                                                        feed_label: ans_batch
                                                    })

            print(img_batch)
            
            # _, loss_value, acc, pred , feed_V0= sess.run([train_op, loss, accuracy, predict, feed_V0],
            #                             feed_dict={
            #                                 feed_img: img_batch,
            #                                 feed_que: que_batch,
            #                                 feed_label: ans_batch
            #                             })
            # print("img_batch")
            # print(img_batch)
            # print(" ")
            # print("feed_V0")
            # print(feed_V0.shape)
            # print(feed_V0)

            train_batch_num += 1
            if train_batch_num % 100 == 0:
                print "Batch: ", train_batch_num, " Loss: ", loss_value, " Learning Rate: ", lr
                train_loss_summary = tf.Summary()
                cost = train_loss_summary.value.add()
                cost.tag = "train_loss"
                cost.simple_value = float(loss_value)
                train_summary_writer.add_summary(train_loss_summary)
            # for test
            # if train_batch_num>100:
            #     assert False
                
        while val_batch_num * args.batch_size < len(qa_data['val']):
            que_batch, ans_batch, img_batch = build_batch(val_batch_num, args.batch_size, \
                                                train_img_feature, img_id_map, qa_data, vocab_data, 'train', args.rnn_size)#really train???
            loss_value, acc, pred = sess.run([loss, accuracy, predict],
                                                    feed_dict={
                                                        feed_img: img_batch,
                                                        feed_que: que_batch,
                                                        feed_label: ans_batch
                                                    })
            val_batch_num += 1
            val_acc_list.append(float(acc))
            val_loss_summary = tf.Summary()
            cost = val_loss_summary.value.add()
            cost.tag = "val_loss"
            cost.simple_value = float(loss_value)
            val_summary_writer.add_summary(val_loss_summary)
        print 'Epoch: ', epoch, ' Accuracy: ', max(val_acc_list)
        val_acc_summary = tf.Summary()
        val_acc = val_acc_summary.value.add()
        val_acc.tag = "val_accuracy"
        val_acc.simple_value = float(acc)
        val_summary_writer.add_summary(val_acc_summary)
        saving = saver.save(sess, os.path.join(args.log_dir, 'model%d.ckpt' % i))
        lr = lr * args.lr_decay

if __name__ == '__main__':
    main()