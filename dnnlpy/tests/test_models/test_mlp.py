import numpy as np
import pytest

import dnnlpy.models.mlp as mlp
import dnnlpy.models.mlp.activation as activation
import dnnlpy.models.mlp.base as base
import dnnlpy.models.mlp.layer as layer
import dnnlpy.models.mlp.loss as loss
import dnnlpy.models.mlp.mlp as model
import dnnlpy.models.mlp.optimizer as optimizer


def test_mlp_public_exports():
    assert mlp.Parameter is base.Parameter
    assert mlp.Module is base.Module
    assert mlp.Optimizer is base.Optimizer
    assert mlp.Flatten is layer.Flatten
    assert mlp.Linear is layer.Linear
    assert mlp.Sigmoid is activation.Sigmoid
    assert mlp.Tanh is activation.Tanh
    assert mlp.ReLU is activation.ReLU
    assert mlp.Softmax is activation.Softmax
    assert mlp.CrossEntropyLoss is loss.CrossEntropyLoss
    assert mlp.MLP is model.MLP
    assert mlp.SGD is optimizer.SGD


def test_parameter_tracks_grad_and_returns_plain_data():
    param = mlp.Parameter([[1.0, 2.0]], dtype=np.float64)
    param.grad = np.array([[0.5, -0.25]])

    result = param + 1.0

    assert isinstance(param, np.ndarray)
    assert not isinstance(param.data, mlp.Parameter)
    assert not isinstance(result, mlp.Parameter)
    assert np.allclose(param.data, np.array([[1.0, 2.0]]))
    assert np.allclose(param.grad, np.array([[0.5, -0.25]]))


def test_flatten_forward_and_backward():
    module = mlp.Flatten()
    x = np.arange(24).reshape(2, 3, 4)

    output = module(x)
    dx = module.backward(np.ones_like(output))

    assert output.shape == (2, 12)
    assert np.array_equal(output[0], np.arange(12))
    assert dx.shape == x.shape


def test_linear_forward_backward_and_parameters():
    module = mlp.Linear(in_features=2, out_features=3)
    module.W[:] = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    module.b[:] = np.array([0.5, -0.5, 1.0])
    x = np.array([[2.0, -1.0], [0.0, 4.0]])
    grad = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]])

    output = module(x)
    dx = module.backward(grad)
    params = list(module.parameters())

    assert np.allclose(output, np.array([[-0.5, -4.5, 3.0], [12.5, -0.5, -3.0]]))
    assert module.W.grad is not None
    assert np.allclose(module.W.grad, np.array([[2.0, 4.0, -2.0], [1.0, -4.0, 13.0]]))
    assert module.b.grad is not None
    assert np.allclose(module.b.grad, np.array([1.5, 1.5, 2.0]))
    assert np.allclose(dx, np.array([[-3.5, 4.0], [3.0, -1.5]]))
    assert params[0] is module.W
    assert params[1] is module.b
    assert 'in_features=2' in repr(module)


def test_linear_backward_requires_forward():
    module = mlp.Linear(in_features=2, out_features=3)

    with pytest.raises(AssertionError, match='forward'):
        module.backward(np.ones((1, 3)))


@pytest.mark.parametrize(
    ('module', 'expected_output', 'expected_grad'),
    [
        (
            mlp.Sigmoid(),
            np.array([[0.26894142, 0.5, 0.73105858]]),
            np.array([[0.19661193, 0.25, 0.19661193]]),
        ),
        (
            mlp.Tanh(),
            np.array([[-0.76159416, 0.0, 0.76159416]]),
            np.array([[0.41997434, 1.0, 0.41997434]]),
        ),
        (
            mlp.ReLU(),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        ),
    ],
)
def test_elementwise_activations_forward_and_backward(
    module: mlp.Module,
    expected_output: np.ndarray,
    expected_grad: np.ndarray,
):
    x = np.array([[-1.0, 0.0, 1.0]])
    grad = np.ones_like(x)

    output = module(x)
    dx = module.backward(grad)

    assert np.allclose(output, expected_output)
    assert np.allclose(dx, expected_grad)


def test_softmax_forward_and_backward():
    module = mlp.Softmax()
    x = np.array([[1.0, 2.0, 3.0]])
    grad = np.array([[0.2, -0.1, 0.4]])

    output = module(x)
    dx = module.backward(grad)

    expected_output = np.array([[0.09003057, 0.24472847, 0.66524096]])
    expected_dot = np.sum(grad * expected_output, axis=1, keepdims=True)
    expected_dx = expected_output * (grad - expected_dot)

    assert np.allclose(output, expected_output)
    assert np.allclose(np.sum(output, axis=1), np.array([1.0]))
    assert np.allclose(dx, expected_dx)


def test_cross_entropy_loss_forward_backward():
    module = mlp.CrossEntropyLoss()
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = np.array([0, 2])

    value = module(logits, targets)
    grad = module.backward()

    probs = activation.softmax(logits)
    expected_value = -np.mean(np.log(probs[np.arange(2), targets] + module.eps))
    expected_grad = probs.copy()
    expected_grad[np.arange(2), targets] -= 1
    expected_grad = expected_grad / 2

    assert np.allclose(value, expected_value)
    assert np.allclose(grad, expected_grad)


def test_cross_entropy_backward_requires_forward():
    module = mlp.CrossEntropyLoss()

    with pytest.raises(AssertionError, match='forward'):
        module.backward()


def test_mlp_forward_backward_yields_recursive_parameter_gradients():
    model = mlp.MLP(input_dim=2, hidden_dim=3, num_classes=2)
    model.fc1.W[:] = np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
    model.fc1.b[:] = np.array([0.0, 0.5, -0.25])
    model.fc2.W[:] = np.array([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
    model.fc2.b[:] = np.array([0.1, -0.2])
    x = np.array([[2.0, -1.0], [0.0, 1.0]])
    grad = np.array([[0.2, -0.3], [0.5, 0.1]])

    logits = model(x)
    dx = model.backward(grad)
    params = list(model.parameters())

    assert logits.shape == (2, 2)
    assert dx.shape == x.shape
    assert len(params) == 4
    assert all(param.grad is not None for param in params)


def test_sgd_updates_parameters_and_zeroes_gradients():
    param = mlp.Parameter([1.0, -2.0])
    skipped = mlp.Parameter([3.0])
    param.grad = np.array([0.5, -0.25])
    optimizer = mlp.SGD([param, skipped], lr=0.1)

    optimizer.step()

    assert np.allclose(param, np.array([0.95, -1.975]))
    assert np.allclose(skipped, np.array([3.0]))

    optimizer.zero_grad(set_to_none=False)
    assert np.allclose(param.grad, np.zeros_like(param))

    optimizer.zero_grad()
    assert param.grad is None
