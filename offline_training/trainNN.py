import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

n_h1=40
n_h2=40
batch_size = 50

def get_batches(x, y, batch_size):
    # this function code is from https://github.com/udacity/deep-learning/tree/master/transfer-learning
    n_batches = len(x)//batch_size #this is a function to get integrate divide
    _, nout = y.shape

    for ii in range(0, n_batches*batch_size, batch_size):
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 
        else:
            X, Y = x[ii:], y[ii:]
            yield X, Y
        # print(Y.shape)
        # Y = Y.reshape((batch_size, nout))
        yield X, Y

def train(args):
    train_x = scipy.io.loadmat('data_time_axis_uniform.mat')['trainX'] #find the data of content of trainX from data.mat
    length, nin = train_x.shape
    train_y = scipy.io.loadmat('data_time_axis_uniform.mat')['trainY']
    _, nout = train_y.shape

    # test_x = scipy.io.loadmat('data_time_fake.mat')['TestX']
    # test_length, _ = test_x.shape
    # test_y = scipy.io.loadmat('data_time_fake.mat')['TestY']

    print(length, nin, nout)
    
    # define neural network
    x_ = tf.placeholder(tf.float32, shape=(None, nin))
    y_ = tf.placeholder(tf.float32, shape=(None, nout))

    fc1 = tf.contrib.layers.fully_connected(x_, n_h1)
    fc2 = tf.contrib.layers.fully_connected(fc1, n_h2)
    fc3 = tf.contrib.layers.fully_connected(fc2, nout, activation_fn=None)
    cost = tf.nn.l2_loss(tf.subtract(fc3, y_))  # mean squared error

    optimizer = tf.train.AdamOptimizer(args.lr).minimize(cost)
    saver = tf.train.Saver()

    loss_log =[]
    epo = []
    error = 0
    with tf.Session() as sess:
        if args.RESUME:
            saver.restore(sess, "checkpoints/" + args.name)
        else:
            sess.run(tf.global_variables_initializer())
        for e in range(args.num_epochs):
            print(e)
            loss_sum = 0
            for x, y in get_batches(train_x, train_y, batch_size):
                feed = {x_: x,
                        y_: y}
                loss, _ = sess.run([cost, optimizer], feed_dict=feed) 
                loss_sum = loss_sum + loss 
            loss_log.append(loss_sum/length)
            print("Epoch: {}/{}".format(e+1, args.num_epochs),
                  "Training loss: {:.5f}".format(loss_sum/length))  # average loss
            biases1 = sess.run('fully_connected/biases:0')
            biases2 = sess.run('fully_connected_1/biases:0')
            biases3 = sess.run('fully_connected_2/biases:0')
            weights1 = sess.run('fully_connected/weights:0')
            weights2 = sess.run('fully_connected_1/weights:0')
            weights3 = sess.run('fully_connected_2/weights:0')
        # y_predict = sess.run(fc3, feed_dict={x_: test_x})
        # cost = sess.run(cost, feed_dict={x_: test_x, y_: test_y})
        # print('cost')
        # print(cost)
        # scipy.io.savemat('Predictions_8.mat', mdict={'Y_predict': y_predict})
        saver.save(sess, "checkpoints/" + args.name)

        if args.PLOT_LEARNING == True:
            plt.plot(range(args.num_epochs), loss_log, 'o-')	
            plt.xlabel('number of epochs')
            plt.ylabel('average loss for each data point')
            plt.savefig('figures/loss.pdf')
            plt.show()
    print(biases1.shape)
    print(weights1.shape)
    print(biases2.shape)
    print(weights2.shape)
    print(biases3.shape)
    print(weights3.shape)
    scipy.io.savemat('axis_uniform/weights1_axis_uniform.mat', mdict={'weights1': weights1})
    scipy.io.savemat('axis_uniform/biases1_axis_uniform.mat', mdict={'biases1': biases1})
    scipy.io.savemat('axis_uniform/weights2_axis_uniform.mat', mdict={'weights2': weights2})
    scipy.io.savemat('axis_uniform/biases2_axis_uniform.mat', mdict={'biases2': biases2})
    scipy.io.savemat('axis_uniform/weights3_axis_uniform.mat', mdict={'weights3': weights3})
    scipy.io.savemat('axis_uniform/biases3_axis_uniform.mat', mdict={'biases3': biases3})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--RESUME', type=bool, default=False)
    parser.add_argument('--PLOT_LEARNING', type = bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--name', type=str, default='train')

    args = parser.parse_args()
    
    train(args)
