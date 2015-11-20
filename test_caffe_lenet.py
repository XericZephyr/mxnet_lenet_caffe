__author__ = 'zhengxu'

import mxnet as mx

def get_caffe_mnist_lenet():

    data = mx.sym.Variable("data")
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
    pool1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    pool2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), stride=(2, 2), pool_type='max')
    flatten = mx.sym.Flatten(data=pool2)
    ip1 = mx.sym.FullyConnected(data=flatten, num_hidden=500)
    relu1 = mx.sym.Activation(data=ip1, act_type='relu')
    ip2 = mx.sym.FullyConnected(data=relu1, num_hidden=10)
    lenet = mx.sym.SoftmaxOutput(data=ip2, name='softmax')

    return lenet


def get_mnist_data(data_dir):

    data_shape = (1, 28, 28)

    batch_size = 3000

    train = mx.io.MNISTIter(
        image=data_dir + "train-images-idx3-ubyte",
        label=data_dir + "train-labels-idx1-ubyte",
        input_shape=data_shape,
        batch_size=batch_size,
        shuffle=True)

    val = mx.io.MNISTIter(
        image=data_dir + "t10k-images-idx3-ubyte",
        label=data_dir + "t10k-labels-idx1-ubyte",
        input_shape=data_shape,
        batch_size=batch_size,
        shuffle=True)

    return (train, val)


def train_mnist(train_data, eval_data, symbol, num_epoch, learning_rate=0.01):

    model = mx.model.FeedForward(
        ctx=[mx.gpu(i) for i in range(3)],
        symbol=symbol,
        num_epoch=num_epoch,
        learning_rate=learning_rate,
        momentum=0.9,
        wd=0.00001
    )

    model.fit(X=train_data,
              eval_data=eval_data,
              batch_end_callback=mx.callback.Speedometer(train_data.batch_size, 10),
              epoch_end_callback=mx.callback.do_checkpoint("model/mnist"),
              work_load_list=[10, 3, 10]
    )

def main():

    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_path = "/home/mxnetdemo/DeepLearning/caffe-fastfpbp/data/mnist/"

    train, val = get_mnist_data(data_path)

    mnist_net = get_caffe_mnist_lenet()

    train_mnist(train, val, mnist_net, 100)

    pass

if __name__ == '__main__':
    main()
