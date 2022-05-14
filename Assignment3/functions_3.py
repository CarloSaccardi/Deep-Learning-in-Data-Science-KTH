import numpy as np

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open('Dataset/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def ComputeCost(X, Y, lambda_, W, b, gamma=None, beta=None, mean=None,var=None,batch_normalization=False):

    # Compute the predictions
    if batch_normalization:
        if mean is None and var is None:
            P, S_BN, S, X_layers, mean, var = \
                EvaluateClassifier(X, W, b, gamma, beta, batch_normalization=True)
        else:
            P, S_BN, S, X_layers = \
                EvaluateClassifier(X, W, b, gamma, beta, mean, var, batch_normalization=True)
    else:
        P, X_layers = EvaluateClassifier(X, W, b)

    # Compute the loss function term
    loss_cross = sum(-np.log((Y*P).sum(axis=0)))

    # Compute the regularization term
    loss_regularization = 0
    for W_l in W:
        loss_regularization += lambda_*((W_l**2).sum())

    # Sum the total cost
    J = loss_cross/X.shape[1]+loss_regularization

    return J

def ComputeGradsNum(X, Y, lambda_, W, b, gamma, beta, mean, var, batch_normalization, h=0.000001):

    # Create lists for saving the gradients by layers
    grad_W = [W_l.copy() for W_l in W]
    grad_b = [b_l.copy() for b_l in b]
    if batch_normalization:
        grad_gamma = [gamma_l.copy() for gamma_l in gamma]
        grad_beta = [beta_l.copy() for beta_l in beta]

    # Compute initial cost and iterate layers k
    c = ComputeCost(X, Y, lambda_, W, b, gamma, beta, mean, var, batch_normalization)
    k = len(W)
    for l in range(k):

        # Gradients for bias
        for i in range(b[l].shape[0]):
            b_try = [b_l.copy() for b_l in b]
            b_try[l][i,0] += h
            c2 = ComputeCost(X, Y, lambda_, W, b_try, gamma, beta, mean, var, batch_normalization)
            grad_b[l][i,0] = (c2-c)/h

        # Gradients for weights
        for i in range(W[l].shape[0]):
            for j in range(W[l].shape[1]):
                W_try = [W_l.copy() for W_l in W]
                W_try[l][i,j] += h
                c2 = ComputeCost(X, Y, lambda_, W_try, b, gamma, beta, mean, var, batch_normalization)
                grad_W[l][i,j] = (c2-c)/h

        if l<(k-1) and batch_normalization:

            # Gradients for gamma
            for i in range(gamma[l].shape[0]):
                gamma_try = [gamma_l.copy() for gamma_l in gamma]
                gamma_try[l][i,0] += h
                c2 = ComputeCost(X, Y, lambda_, W, b, gamma_try, beta, mean, var, batch_normalization)
                grad_gamma[l][i,0] = (c2-c)/h

            # Gradients for betas
            for i in range(beta[l].shape[0]):
                beta_try = [beta_l.copy() for beta_l in beta]
                beta_try[l][i,0] += h
                c2 = ComputeCost(X, Y, lambda_, W, b, gamma, beta_try, mean, var, batch_normalization)
                grad_beta[l][i,0] = (c2-c)/h

    if batch_normalization:
        return grad_W, grad_b, grad_gamma, grad_beta
    else:
        return grad_W, grad_b

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()
