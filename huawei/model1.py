from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import tensorflow as tf

tf.flags.DEFINE_integer('max_steps', 1000, 'number of training iterations.')
tf.flags.DEFINE_string('data_url', None, 'dataset directory.')
tf.flags.DEFINE_string('train_url', None, 'saved model directory.')
print(os.getcwd())
FLAGS = tf.flags.FLAGS

batch_size = 200
num_steps = 100
def main(*args):
    # Train model
    work_directory = FLAGS.data_url

    filename = 'train_set.csv'
    print('Training model...')
    filepath = os.path.join(work_directory, filename)
    print(filepath)
    data = pd.read_csv(filepath)
    sess = tf.InteractiveSession()
    X_train = data.iloc[:900000,1:-1].copy()
    X_train = (X_train-X_train.mean())/X_train.std()
    X_train.fillna(0.1,inplace=True)
    X_train = X_train.values
    y_train = data.iloc[:900000,-1].copy()
    n_inputs = X_train.shape[1]
    x = tf.placeholder(tf.float32,[None,n_inputs])
    y = tf.placeholder(tf.float32,[None,1])

    # 定义神经网络中间层
    Weights_L1 = tf.Variable(tf.random_normal([n_inputs,100]))
    # biases_L1 = tf.Variable(tf.zeros([n_inputs,1]))
    Wx_plus_b_L1 = tf.matmul(x,Weights_L1)
    L1 = tf.nn.tanh(Wx_plus_b_L1)

    # 定义神经网络输出层
    Weights_L2 = tf.Variable(tf.random_normal([100,1]))
    # biases_L2 = tf.Variable(tf.zeros([1,1]))
    Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)
    prediction = tf.nn.sigmoid(Wx_plus_b_L2)
    loss = tf.reduce_mean(tf.square(y-prediction))
    # 定义一个梯度下降法来进行训练的优化器 学习率0.1
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    tf.summary.scalar('loss', loss)
    for step in range(40):
        for i in range(1, num_steps + 1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size,:]
            batch_y = y_train[i*batch_size:(i+1)*batch_size,:]
            _,l = sess.run([train_step,loss],feed_dict={x:batch_x,y:batch_y})
        if step%10 == 0:
            print('.')

    print('Done training!')

    
    builder = tf.saved_model.builder.SavedModelBuilder('model')

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

    prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_images':
            prediction_signature,
    },
    main_op=tf.tables_initializer(),
    strip_default_attrs=True)

    builder.save()
    print('Done exporting!')

if __name__ == '__main__':
    tf.app.run(main=main)