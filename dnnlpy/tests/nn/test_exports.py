import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


def test_nn_exports_new_modules():
    for name in [
        'Identity',
        'Linear',
        'Sigmoid',
        'Tanh',
        'ReLU',
        'GELU',
        'Softmax',
        'LogSoftmax',
        'CrossEntropyLoss',
        'Dropout',
        'Dropout1d',
        'Dropout2d',
        'Dropout3d',
    ]:
        assert hasattr(dnn, name)


def test_functional_exports_new_functions():
    for name in [
        'linear',
        'sigmoid',
        'tanh',
        'relu',
        'gelu',
        'softmax',
        'log_softmax',
        'dropout',
        'dropout1d',
        'dropout2d',
        'dropout3d',
        'naive_attention',
        'scaled_dot_product_attention',
        'multi_head_attention',
        'generate_causal_mask',
        'cross_entropy',
    ]:
        assert hasattr(dF, name)
