import torch

from dnnl.models.vae import VAE, AutoEncoder


def test_autoencoder_encode_decode_and_forward_shapes():
    model = AutoEncoder((1, 4, 4), hidden_dim=8, latent_dim=3)
    x = torch.rand(2, 1, 4, 4)

    z = model.encode(x)
    x_hat = model.decode(z)
    output = model(x)

    assert z.shape == (2, 3)
    assert x_hat.shape == x.shape
    assert output.shape == x.shape
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_vae_forward_and_loss_shapes():
    model = VAE((1, 4, 4), hidden_dim=8, latent_dim=3)
    x = torch.rand(2, 1, 4, 4)

    x_hat, mu, logvar = model(x)
    loss, recon_loss, kl_loss = VAE.loss(x_hat, x, mu, logvar)

    assert x_hat.shape == x.shape
    assert mu.shape == (2, 3)
    assert logvar.shape == (2, 3)
    assert loss.ndim == recon_loss.ndim == kl_loss.ndim == 0
    assert torch.isfinite(loss)


def test_vae_loss_supports_mse_and_rejects_unknown_loss():
    x = torch.zeros(2, 1, 2, 2)
    x_hat = torch.ones_like(x) * 0.5
    mu = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)

    loss, recon_loss, kl_loss = VAE.loss(x_hat, x, mu, logvar, loss_fn='mse', beta=0.5)

    assert torch.allclose(loss, recon_loss)
    assert torch.allclose(kl_loss, torch.tensor(0.0))

    try:
        VAE.loss(x_hat, x, mu, logvar, loss_fn='mae')  # type: ignore[arg-type]
    except NotImplementedError as exc:
        assert 'Unsupported loss function' in str(exc)
    else:
        raise AssertionError('`VAE.loss` should reject unsupported loss functions.')
