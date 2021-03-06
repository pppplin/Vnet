import tensorflow as tf
import numpy as np
import data_loader
import argparse
import os
import h5py
import time

import image_processor
import answer_generator

def load_VGG_feature(data_dir, split):
    VGG_feature = None
    img_id_list = None
    with h5py.File(os.path.join(data_dir, split, (split + '_debug_vgg16.h5')), 'r') as hf:
        VGG_feature = np.array(hf.get('fc7_feature'))
    with h5py.File(os.path.join(data_dir, split, (split + '_debug_img_id.h5')), 'r') as hf:
        img_id_list = np.array(hf.get('img_id'))
    return VGG_feature, img_id_list
def VGG_16_extract(split, args):
    # If the feature is already recorded
    if os.path.exists(os.path.join(args.data_dir, split, split + '_debug_vgg16.h5')):
        print 'Image Feature Data Calculated. Start Loading Feature Data...'
        return load_VGG_feature(args.data_dir, split)
    train_img_feature, train_img_id_list = image_processor.VGG_16_extract('train', args)
    print 'Writing Debug Data'
    hf5_fc7 = h5py.File(os.path.join(args.data_dir, split, split + '_debug_vgg16.h5'), 'w')
    hf5_fc7.create_dataset('fc7_feature', data=train_img_feature[:128,:])
    hf5_fc7.close()

    hf5_img_id = h5py.File(os.path.join(args.data_dir, split, split + '_debug_img_id.h5'), 'w')
    hf5_img_id.create_dataset('img_id', data=train_img_id_list[:128])
    hf5_img_id.close()
    print 'Image Information Encoding Done'
    return data_loader.load_VGG_feature(args.data_dir, split)

def right_align(seq, length):
    mask = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        mask[i][N - length[i]:N - 1] = seq[i][0:length[i] - 1]
    return mask

def build_batch(batch_num, batch_size, img_feature, img_id_map, qa_data, vocab_data, split):
    qa = qa_data[split]
    batch_start = (batch_num * batch_size) % len(qa)
    batch_end = min(len(qa), batch_start + batch_size)
    size = batch_end - batch_start
    sentence = np.ndarray((n, vocab_data['max_que_length']), dtype='int32')
    answer = np.zeros((n, len(vocab_data['ans_vocab'])))
    img = np.ndarray((n, 4096))

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
    parser.add_argument('--data_dir', type=str, default='data', help='Data Directory')
    parser.add_argument('--log_dir', type=str, default='log', help='Checkpoint File Directory')
    parser.add_argument('--top_num', type=int, default=1000, help='Top Number Answer')

    parser.add_argument('--batch_size', type=int, default=64, help='Image Training Batch Size')
    parser.add_argument('--num_output', type=int, default=1000, help='Number of Output')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout Rate')
    parser.add_argument('--init_bound', type=float, default=0.8, help='Parameter Initialization Distribution Bound')

    parser.add_argument('--hidden_dim', type=int, default=1024, help='RNN Hidden State Dimension')
    parser.add_argument('--rnn_size', type=int, default=512, help='Size of RNN Cell')
    parser.add_argument('--rnn_layer', type=int, default=2, help='Number of RNN Layers')
    parser.add_argument('--que_embed_size', type=int, default=200, help='Question Embedding Dimension')

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning Rate Decay Factor')
    parser.add_argument('--num_iteration', type=int, default=15000, help='Number of Training Iterations')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of Training Epochs')
    parser.add_argument('--grad_norm', type=int, default=5, help='Maximum Norm of the Gradient')
    args = parser.parse_args()


    print 'Reading Question Answer Data'
    qa_data, vocab_data = data_loader.load_qa_data(args.data_dir, args.top_num)
    train_img_feature, train_img_id_list = VGG_16_extract('train', args)

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
        'max_que_length': vocab_data['max_que_length'],
        'num_output': args.num_output,
        'dropout_rate': args.dropout_rate,
        'data_dir': args.data_dir,
        'top_num': args.top_num,
        'init_bound': args.init_bound
        })
    loss, feed_img, feed_que, feed_label = generator.train_model()
    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    sess = tf.Session()

    tf.initialize_all_variables().run()

    print 'Training Start...'
    saver = tf.train.Saver()
    for epoch in range(args.num_epoch):
        batch_num = 0
        while batch_num * args.batch_size < len(qa_data['train']):
            que_batch, ans_batch, img_batch = build_batch(batch_num, args.batch_size, \
                                                train_img_feature, img_id_map, qa_data, vocab_data, 'train')
            _, loss_value = sess.run([train_op, loss],
                                                    feed_dict={
                                                        feed_img: img_batch,
                                                        feed_que: que_batch,
                                                        feed_label: ans_batch
                                                    })
            batch_num += 1
        saving = saver.save(sess, os.path.join(args.log_dir, 'Models', 'model%d.ckpt' % i))
        new_lr = lr * args.lr_decay

if __name__ == '__main__':
    main()