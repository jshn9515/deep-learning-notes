import inspect

import numpy as np
import pytest

import dnnl.models.mlp.activation as activation
import dnnl.models.mlp.base as base
import dnnl.models.mlp.linear as linear
import dnnl.models.mlp.loss as loss
from dnnl.models import mlp


def test_mlp_modules_have_docstrings():
    modules = [base, linear, activation, loss]

    for module in modules:
        for name in module.__all__:
            member = getattr(module, name)
            assert inspect.getdoc(member), name

            if inspect.isclass(member):
                for method_name, method in inspect.getmembers(
                    member,
                    inspect.isfunction,
                ):
                    if method.__qualname__.startswith(f'{member.__name__}.'):
                        assert inspect.getdoc(method), f'{name}.{method_name}'


def test_mlp_public_exports():
    assert mlp.Module is base.Module
    assert mlp.Linear is linear.Linear
    assert mlp.sigmoid is activation.sigmoid
    assert mlp.Sigmoid is activation.Sigmoid
    assert mlp.tanh is activation.tanh
    assert mlp.Tanh is activation.Tanh
    assert mlp.relu is activation.relu
    assert mlp.ReLU is activation.ReLU
    assert mlp.softmax is activation.softmax
    assert mlp.Softmax is activation.Softmax
    assert mlp.cross_entropy is loss.cross_entropy
    assert mlp.CrossEntropyLoss is loss.CrossEntropyLoss


def test_linear_forward_backward_and_parameters():
    module = mlp.Linear(in_features=2, out_features=3)
    module.W = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    module.b = np.array([0.5, -0.5, 1.0])
    x = np.array([[2.0, -1.0], [0.0, 4.0]])
    grad = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]])

    output = module.forward(x)
    dx = module.backward(grad)

    assert np.allclose(
        output,
        np.array([[-0.5, -4.5, 3.0], [12.5, -0.5, -3.0]]),
    )
    assert module.dW is not None
    assert np.allclose(
        module.dW,
        np.array([[2.0, 4.0, -2.0], [1.0, -4.0, 13.0]]),
    )
    assert module.db is not None
    assert np.allclose(module.db, np.array([1.5, 1.5, 2.0]))
    assert np.allclose(
        dx,
        np.array([[-3.5, 4.0], [3.0, -1.5]]),
    )
    assert module.parameters() == [module.W, module.b]


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
    module,
    expected_output,
    expected_grad,
):
    x = np.array([[-1.0, 0.0, 1.0]])
    grad = np.ones_like(x)

    output = module.forward(x)
    dx = module.backward(grad)

    assert np.allclose(output, expected_output)
    assert np.allclose(dx, expected_grad)


def test_softmax_forward_and_backward():
    module = mlp.Softmax()
    x = np.array([[1.0, 2.0, 3.0]])
    grad = np.array([[0.2, -0.1, 0.4]])

    output = module.forward(x)
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

    value = module.forward(logits, targets)
    grad = module.backward()

    probs = mlp.softmax(logits)
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
