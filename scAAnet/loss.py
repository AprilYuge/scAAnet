import numpy as np
import tensorflow as tf

def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.math.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.math.divide(tf.math.reduce_sum(x), nelem)

def poisson_loss(y_true, y_pred, mean=True, masking=False):
    
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    y_true = _nan2zero(y_true)

    ret = y_pred - y_true*tf.math.log(y_pred+1e-10) + tf.math.lgamma(y_true+1.0)
    
    if mean:
        if masking:
            ret = _reduce_mean(ret)
        else:
            ret = tf.math.reduce_mean(ret)

    return ret

def ZIPoisson_loss(y_true, outputs, ridge_lambda=0.0,
              mean=True, masking=False, scale_factor=1.0, debug=False):
    
    y_pred, pi = outputs
    softplus_pi = tf.math.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)

    # reuse existing Poisson neg.log.lik.
    # mean is always False here, because everything is calculated
    # element-wise. we take the mean only in the end
    poisson_case = poisson_loss(y_true, y_pred, mean=False) - (-softplus_pi - pi)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor

    zero_case = -tf.math.softplus(-pi - y_pred) + softplus_pi
    result = tf.where(tf.math.less(y_true, 1e-8), zero_case, poisson_case)
    ridge = ridge_lambda*tf.math.square(pi)
    result += ridge

    if mean:
        if masking:
            result = _reduce_mean(result)
        else:
            result = tf.math.reduce_mean(result)

    result = _nan2inf(result)

    if debug:
        tf.summary.histogram('poisson_case', poisson_case)
        tf.summary.histogram('zero_case', zero_case)
        tf.summary.histogram('ridge', ridge)

    return result

def NB_loss(y_true, outputs, mean=True,
            masking=False, scale_factor=1.0, debug=False):
    eps = 1e-10
    y_pred, theta = outputs

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor

    if masking:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)

    # Clip theta
    theta = tf.math.minimum(theta, 1e6)

    t1 = tf.math.lgamma(theta+eps) + tf.math.lgamma(y_true+1.0) - tf.math.lgamma(y_true+theta+eps)
    t2 = (theta+y_true) * tf.math.log(1.0 + (y_pred/(theta+eps))) + (y_true * (tf.math.log(theta+eps) - tf.math.log(y_pred+eps)))

    if debug:
        assert_ops = [
            tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
            tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
            tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]

        tf.summary.histogram('t1', t1)
        tf.summary.histogram('t2', t2)

        with tf.control_dependencies(assert_ops):
            final = t1 + t2

    else:
        final = t1 + t2

    final = _nan2inf(final)

    if mean:
        if masking:
            final = tf.math.divide(tf.math.reduce_sum(final), nelem)
        else:
            final = tf.math.reduce_mean(final)


    return final

def ZINB_loss(y_true, outputs, ridge_lambda=0.0,
              mean=True, masking=False, scale_factor=1.0, debug=False):
    eps = 1e-10
    y_pred, theta, pi = outputs
    softplus_pi = tf.math.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)

    # reuse existing NB neg.log.lik.
    # mean is always False here, because everything is calculated
    # element-wise. we take the mean only in the end
    nb_case = NB_loss(y_true, [y_pred, theta], mean=False) - (-softplus_pi - pi)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.math.minimum(theta, 1e6)

    pi_theta_log = -pi + theta * (tf.math.log(theta+eps) - tf.math.log(theta+y_pred+eps))
    zero_case = -tf.math.softplus(pi_theta_log) + softplus_pi
    
    result = tf.where(tf.math.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda*tf.math.square(pi)
    result += ridge

    if mean:
        if masking:
            result = _reduce_mean(result)
        else:
            result = tf.math.reduce_mean(result)

    result = _nan2inf(result)

    if debug:
        tf.summary.histogram('nb_case', nb_case)
        tf.summary.histogram('zero_case', zero_case)
        tf.summary.histogram('ridge', ridge)

    return result