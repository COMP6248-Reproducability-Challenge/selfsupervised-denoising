from torch.distributions.poisson import Poisson

def add_gaussian_noise(tensor_img, std_dev, mean=0):
    """
    Add Gaussian noise to an image.
    We want to add zero-mean noise by default (there is no need to
    change the mean colour of the image).
    """
    return tensor_img + torch.randn(tensor_img.size()) * (std_dev / 255) + mean


def add_poisson_noise(tensor_img, lam):
    """
    Add Poisson noise to an image.
    """
    dist = Poisson(lam)
    noise = dist.sample(tensor_img.size())
    #return (tensor_img + noise) / lam
    raise NotImplementedError()

def add_binomial_noise():
    """
    Add Binomial (aka Multiplicative Bernoulli) noise to an image.
    """
    raise NotImplementedError()

def add_impulse_noise():
    """
    Add Impulse noise to an image.
    """
    raise NotImplementedError()