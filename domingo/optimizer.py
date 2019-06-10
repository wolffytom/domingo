class Optimizer(object):
    def __init__(self):
        pass

class SGD_Optimizer(Optimizer):
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate

    def update(self, w, g): 
        w[:] = w - self.learning_rate * g
        return True

class RMSProp_Optimizer(Optimizer):
    def __init__(self, learning_rate=0.0001, epsilon=1e-10, decay=0.9):
        self.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay = decay

    def update(self, w, r, g):
        assert w.shape == r.shape == g.shape
        learning_rate = self.learning_rate
        epsilon = self.epsilon
        decay = self.decay
        r[:] = r * decay + (1-decay) * (g * g)
        w[:] = w - learning_rate * (g / (epsilon + np.sqrt(r)))
        return True

class Ftrl_Optimizer(Optimizer):
    def __init__(self, alpha=0.5, beta=1.0, L1=1.0, L2=1.0):
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

    def get_w(self, n, z):
        L1 = self.L1
        L2 = self.L2
        beta = self.beta
        alpha = self.alpha
        sign = np.sign(z)
        w = (sign * L1 - z) / ((beta + np.sqrt(n)) / alpha + L2)
        w[sign * z <= L1] = 0
        return w

    def update(self, n, z, g):
        assert n.shape == z.shape == g.shape
        sigma = (np.sqrt(n + g * g) - np.sqrt(n)) / self.alpha
        z += g - sigma * self.get_w(n, z)
        n += g * g
        return True

    def get_max_like_y(self, probs):
        output_num = self.params['output_num']
        assert probs.shape == (output_num,)
        return np.argmax(probs)
