__author__ = 'zhengxu'

import mxnet as mx


def test_symbol():

    A = mx.sym.Variable("A")
    B = mx.sym.Variable("B")
    C = A + B

    a = mx.nd.array([1, 3])
    b = mx.nd.array([2, 4])

    C_exec = C.bind(ctx=mx.cpu(), args={'A': a, 'B': b})

    C_exec.forward()

    pass


if __name__ == '__main__':
    test_symbol()