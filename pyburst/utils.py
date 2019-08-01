import numpy

def pseudoinverse(a, rcond=1e-15):
    """
    Compute pseudoinverse of 2D array corresponding to the last two dimensions
    of the input array
    """
    swap = numpy.arange(a.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]

    u, s, v = numpy.linalg.svd(a)
    cutoff = numpy.maximum.reduce(s, axis=-1, keepdims=True) * rcond

    mask = s > cutoff
    s[mask] = 1. / s[mask]
    s[~mask] = 0

    return numpy.einsum('...uv,...vw->...uw',
                     numpy.transpose(v, swap) * s[..., None, :],
                     numpy.transpose(u, swap))
