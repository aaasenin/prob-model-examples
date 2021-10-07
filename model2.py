from scipy.signal import fftconvolve
import numpy as np

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, _ = X.shape
    h, w = F.shape

    from_norm_denom = H * W * (-np.log(s * np.sqrt(2 * np.pi)))
    under_square = fftconvolve(X, F.reshape(h, w, 1)[::-1, ::-1, :], mode='valid')
    XB = X * B.reshape(H, W, 1)
    under_square -= fftconvolve(XB, np.ones((h, w, 1)), mode='valid')
    under_square += np.sum(XB, axis=(0, 1)).reshape(1, 1, -1)
    X_summed = np.sum(X ** 2, axis=(0, 1)).reshape(1, 1, -1)
    F_summed = np.sum(F ** 2)
    B_squared = B ** 2
    B_summed = np.sum(B_squared, axis=(0, 1))
    B_convolved = fftconvolve(B_squared.reshape(H, W, 1), np.ones((h, w, 1)), mode='valid')
    under_square *= -2
    under_square += X_summed + F_summed + B_summed - B_convolved
    result = -1 / (2 * s ** 2) * under_square + from_norm_denom

    return result

def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    X_d_theta_logged = calculate_log_probability(X, F, B, s)

    if use_MAP == False:
        left = np.sum(X_d_theta_logged * q)
        A_volumed = A.reshape(A.shape[0], A.shape[1], 1)
        middle = np.sum(np.log(A_volumed) * q)
        non_zero_q = q > 0
        right = np.sum(np.log(q[non_zero_q]) * q[non_zero_q])

        return left + middle - right

    if use_MAP == True:
        left = np.sum(X_d_theta_logged[q[0], q[1], np.arange(q.shape[1])])
        middle = np.sum(np.log(A)[q[0], q[1]])

        return left + middle

def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """

    X_d_theta_logged = calculate_log_probability(X, F, B, s)
    A_volumed = A.reshape(A.shape[0], A.shape[1], 1)

    nom_logged = X_d_theta_logged + np.log(A_volumed)
    max_logged = np.max(nom_logged, axis=(0, 1))
    nom = np.exp(nom_logged - max_logged)
    denom = np.sum(nom, axis=(0, 1))

    probs = nom / (denom.reshape(1, 1, -1) + 1e-50)

    if use_MAP == False:
        return probs

    if use_MAP == True:
        res = np.zeros((2, X.shape[2]))
        for k in range(X.shape[2]):
            res[:, k] = np.unravel_index(np.argmax(probs[:, :, k]), A.shape)

        return res.astype(int)

def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    if use_MAP == False:
        F = fftconvolve(X, q[::-1, ::-1, ::-1], mode='valid') / K

        q_convolved = fftconvolve(q, np.ones((h, w, 1)), mode='full')
        common = 1 - q_convolved
        B = np.sum(common * X, axis=2) / np.sum(common, axis=2)

        under_square = fftconvolve(X, F.reshape(h, w, 1)[::-1, ::-1, :], mode='valid')
        XB = X * B.reshape(H, W, 1)
        under_square -= fftconvolve(XB, np.ones((h, w, 1)), mode='valid')
        under_square += np.sum(XB, axis=(0, 1)).reshape(1, 1, -1)
        X_summed = np.sum(X ** 2, axis=(0, 1)).reshape(1, 1, -1)
        F_summed = np.sum(F ** 2)
        B_squared = B ** 2
        B_summed = np.sum(B_squared, axis=(0, 1))
        B_convolved = fftconvolve(B_squared.reshape(H, W, 1), np.ones((h, w, 1)), mode='valid')
        under_square *= -2
        under_square += X_summed + F_summed + B_summed - B_convolved

        s_nom = np.sum(under_square * q)
        s_denom = K * X.shape[0] * X.shape[1]
        s_2 = s_nom / s_denom

        A = np.sum(q, axis=2) / K

        return F.reshape(F.shape[0], F.shape[1]), B, np.sqrt(s_2), A

    if use_MAP == True:
        F = np.zeros((h, w))
        B = np.zeros((H, W))
        back_mask_denom = np.zeros((H, W))
        A = np.zeros((H - h + 1, W - w + 1))

        for k in range(K):
            h_idxs = np.arange(q[0, k], q[0, k] + h).astype(int)
            w_idxs = np.arange(q[1, k], q[1, k] + w).astype(int)
            F += np.sum(X[np.ix_(h_idxs, w_idxs, np.array([k]))], axis=2)

            background_mask = np.ones((H, W)) * True
            background_mask[np.ix_(h_idxs, w_idxs)] = False
            B += X[:,:,k] * background_mask
            back_mask_denom += background_mask

            A[q[0, k], q[1, k]] += 1

        # potencial bug
        F /= K
        B /= (back_mask_denom + 1e-50)
        A /= K

        s_2 = 0

        for k in range(K):
            h_idxs = np.arange(q[0, k], q[0, k] + h)
            w_idxs = np.arange(q[1, k], q[1, k] + w)

            sub_image = np.copy(B)
            sub_image[np.ix_(h_idxs, w_idxs)] = F
            s_2 += np.sum((X[:, :, k] - sub_image) ** 2)

        s_2 /= K * H * W

        return F, B, np.sqrt(s_2), A

def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step);
        number_of_iters is actual number of iterations that was done.
    """
    H, W, K = X.shape

    if A is None:
        A = np.ones((H - h + 1, W - w + 1))
        A /= np.sum(A)
    if F is None:
        F = np.abs(np.random.rand(h, w))
    if B is None:
        B = np.abs(np.random.rand(H, W))
    if s is None:
        s = 1

    iter_idx = 0
    curr_ll = -np.inf
    prev_ll = None
    LL = []
    while (iter_idx < max_iter):
        A_zeros = A == 0
        A[A_zeros] = 1e-12

        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)

        A_zeros = A == 0
        A[A_zeros] = 1e-12

        prev_ll = curr_ll
        curr_ll = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        LL.append(curr_ll)
        if (curr_ll - prev_ll < tolerance):
            break
        iter_idx += 1
    return F, B, s, A, np.array(LL)

def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    lowest_L = -np.inf
    lowest_F, lowest_B, lowest_s, lowest_A = None, None, None, None
    for iter in range(n_restarts):
        F, B, s, A, LL = run_EM(X, h, w, tolerance, max_iter, use_MAP)
        if LL[-1] > lowest_L:
            lowest_F, lowest_B, lowest_s, lowest_A, lowest_L = F, B, s, A, LL[-1]

    return lowest_F, lowest_B, lowest_s, lowest_A, lowest_L
