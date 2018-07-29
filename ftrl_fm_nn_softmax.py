#coding:utf-8
import unittest, logging, time, json, random, math
from math import log, sqrt
import numpy as np
from collections import defaultdict

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

class Base_Model():
    def __init__(self):
        if 'params' not in dir(self):
            self.params = {}
        if 'vars' not in dir(self):
            self.vars = {} 
        if 'sparse_vars' not in dir(self):
            self.sparse_vars = {}

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Fid_Eliminate_Model(Base_Model):
    def __init__(self, fid_eliminate, fid_expire_ts=3600*24*7):
        Base_Model.__init__(self)
        self.params['fid_eliminate'] = fid_eliminate
        self.params['fid_expire_ts'] = fid_expire_ts
        self.vars['fid_eliminate_time'] = defaultdict(int)
        if 'load_model_opts' not in dir(self):
            self.load_model_opts = []

    def update_eliminate_time(self, x):
        for xi in x:
            self.fid_eliminate_time[xi] = int(time.time() + self.fid_expire_ts)
        return True

    def dump_model(self):
        self.do_feature_eliminate()

        data = {
                'params' : self.params,
                'vars' : self.vars,
                'sparse_vars' : self.sparse_vars
                }
        data_json = json.dumps(data, cls=NumpyEncoder)
        return data_json

    def load_model(self, data_str):
        data = json.loads(data_str)
        assert data.has_key('params')
        assert data.has_key('vars')
        assert data.has_key('sparse_vars')
        self.params = data['params']
        self.vars = data['vars']
        self.sparse_vars = data['sparse_vars']
        for opt in self.load_model_opts:
            assert opt()
        return True

    def do_feature_eliminate(self):
        fid_eliminate = self.params['fid_eliminate']
        if fid_eliminate == False:
            return True
        fid_eliminate_time = self.vars['fid_eliminate_time']
        now = time.time()
	timeout_fids = []
        fids = fid_eliminate_time.keys()
        fid_eliminate_time_list = fid_eliminate_time.values()
        for i in range(len(fid_eliminate_time)):
            if now > fid_eliminate_time_list[i]:
                timeout_fids.append(fids[i])
        for variable_dict in self.vars.values():
            for fid in timeout_fids:
                del variable_dict[fid]
        return True

class Layer(object):
    def __init__(self, name, model):
        self.name = name
        self._model = model

class Sparse_Softmax_Cross_Entropy_Layer(Layer):
    def __init__(self, name, model, input_output_width):
        Layer.__init__(self, name, model)
        self._input_output_width = input_output_width

    def flow(self, input_tuple):
        self._input_, self._input_gradient = input_, input_gradient = input_tuple
        assert input_.shape == input_gradient.shape == (self._input_output_width, )
        self._output_ = output_ = np.zeros(self._input_output_width)
        return output_

    def forward(self):
        input_ = self._input_
        output_ = self._output_

        input_min = np.min(input_)
        #output_[:] = np.exp(input_ - input_min)
        output_[:] = np.exp(input_)
        output_[:] = output_ / np.sum(output_)
        
        max_likelihood_y = np.argmax(output_)
        self._model.placeholders['max_likelihood_y'] = max_likelihood_y
        return True

    def backward(self):
        assert 'y' in self._model.placeholders
        output_ = self._output_
        input_gradient = self._input_gradient
        y_ = self._model.placeholders['y']
        assert 0 <= y_ and y_ < self._input_output_width and type(y_) == int
        # calculate cross entropy for maintain
        py_ = output_[y_]
        py_ = max(min(py_, 1. - 10e-15), 10e-15)
        cross_entropy_ = -log(py_)
        self._model.placeholders['cross_entropy'] = cross_entropy_
        # calcaulate gradient
        input_gradient[:] = output_
        input_gradient[y_] -= 1.0
        return True

class Concat_Layer(Layer):
    def __init__(self, name, model):
        Layer.__init__(self, name, model)

    def flow(self, input1_tuple, input2_tuple):
        self._input1_, self._input1_gradient = input1_, input1_gradient = input1_tuple
        self._input2_, self._input2_gradient = input2_, input2_gradient = input2_tuple
        self._output_ = output_ = np.concatenate( input1_, input2_ )
        self._output_gradient = output_gradient = np.concatenate( input1_gradient, input2_gradient )
        return (output_, output_gradient)

class Sparse_Vector_Embedding_layer(Layer):
    def __init__(self,  name, model, slotslist,  vectorsize=8, optimizer = SGD_Optimizer(), output_type = 'flat'):
        Layer.__init__(self, name, model)
        params = self._model.params
        self._slotsdict = params[name + '_slotsdict'] = self._get_slotsdict(slotslist)
        self._optimizer = optimizer
        self._vectorsize = params[name + '_vectorsize'] = vectorsize
        self._output_type = output_type

        self._varnames = params[name + '_varnames'] = {
                SGD_Optimizer : [name + '_w'],
                RMSProp_Optimizer : [name + '_w', name + '_wr'],
                Ftrl_Optimizer : [name + '_n', name + '_z'],
                }[ type(self._optimizer) ]
        assert self._init_vars()

        self._model.load_model_opts.append(self._load_variables)

    def flow(self):
        slotsdict = self._slotsdict
        slots_num = len(slotsdict)
        vectorsize = self._vectorsize
        output_shape = {
                'flat' : (slots_num * vectorsize, ),
                'mat' : (slots_num , vectorsize )
                }[self._output_type]
        self._output_ = output_ = np.zeros(output_shape)
        self._output_gradient = output_gradient = np.zeros(output_shape)
        return (output_, output_gradient)

    def _get_slotsdict(self, slotslist):
        slotslist.sort()
        slotsdict = {}
        for i in range(len(slotslist)):
            slotsdict[slotslist[i]] = i
        return slotsdict

    def _init_vars(self):
        for varname in self._varnames:
            self._model.sparse_vars[varname] = defaultdict(lambda : np.zeros(self._vectorsize))
        return True

    def get_output_width(self):
        return len(self._slotsdict) * self._vectorsize

    def _load_variables(self):
        embedded_size_per_slot = self.params['embedded_size_per_slot']
        for varname in self._varnames:
            v = self._model.sparse_vars[varname]
            vnew = defaultdict(lambda : np.zeros(self._vectorsize))
            for k,v in v.iteritems():
                vnew[k] = np.array(v, dtype=float)
            self._model.sparse_vars[varname] = vnew
        return True

    def _get_slot(self, fid):
        return long(fid) / long(math.pow(2,54)) # this will cause error in python3

    def _init_slots_dict(self):
        name = self.name
        slotslist = self.params[name + '_slotslist']
        slotsdict = {}
        for i in range(len(slotslist)):
            slotsdict[slotslist[i]] = i
        self._slotsdict = slotsdict
        return True

    def forward(self):
        x_ = self._model.placeholders['x']
        slotsdict = self._slotsdict
        slots_num = len(slotsdict)
        vectorsize = self._vectorsize
        mat_output_ = self._output_.reshape( slots_num, vectorsize )
        self._x_fid_slot_dict = x_fid_slot_dict = {}
        for fid in x_:
            slot = self._get_slot(fid)
            if slot not in slotsdict:
                logging.warning( 'slot %s not in model slotsdict' % str(slot) )
            else:
                x_fid_slot_dict[fid] = slotsdict[slot]

        sparse_vars = self._model.sparse_vars
        if '_get_w' not in dir(self):
            name = self.name
            self._get_w = {
                SGD_Optimizer : (
                    lambda fid : sparse_vars[name+'_w'][fid] ),
                RMSProp_Optimizer : (
                    lambda fid : sparse_vars[name+'_w'][fid] ),
                Ftrl_Optimizer : (
                    lambda fid : self._optimizer.get_w(
                        sparse_vars[name+'_n'][fid],
                        sparse_vars[name+'_z'][fid]) )
                } [ type(self._optimizer) ]
        get_w = self._get_w

        mat_output_.fill(0.)
        for fid, slotidx in x_fid_slot_dict.iteritems():
            stranger_fid = False
            for varname in self._varnames:
                var_ = sparse_vars[varname]
                if not var_.has_key(fid):
                    stranger_fid = True
                    break
            if not stranger_fid:
                mat_output_[slotidx] += get_w(fid)
        return True

    def backward(self):
        slots_num = len(self._slotsdict)
        mat_output_gradient = self._output_gradient.reshape(
                slots_num , self._vectorsize )
        x_fid_slot_dict = self._x_fid_slot_dict
        if '_parse_vars_update_opt_' not in dir(self):
            name = self.name
            update_func = self._optimizer.update
            sparse_vars = self._model.sparse_vars
            self._parse_vars_update_opt_ = {
                SGD_Optimizer : (
                    lambda fid, gradient : \
                            update_func(
                                sparse_vars[name+'_w'][fid],
                                gradient) 
                                ) ,
                RMSProp_Optimizer : (
                    lambda fid, gradient : \
                            update_func(
                                sparse_vars[name+'_w'][fid],
                                sparse_vars[name+'_wr'][fid],
                                gradient)
                                ) ,
                Ftrl_Optimizer : (
                    lambda fid, gradient : \
                            update_func(
                                sparse_vars[name+'_n'][fid],
                                sparse_vars[name+'_z'][fid],
                                gradient)
                                ) ,
                } [ type(self._optimizer) ]
        parse_vars_update_opt_ = self._parse_vars_update_opt_

        for fid, slotidx in x_fid_slot_dict.iteritems():
            fid_gradient = mat_output_gradient[slotidx]
            assert True == parse_vars_update_opt_( fid, fid_gradient )
        return True

class FM_Layer(Layer):
    def __init__(self, name, model,
            input_size,
            fm_k, # input shape is (input_size, fm_k)
            fm_init_scale=1.0,
            ):
        Layer.__init__(self, name, model)
        self._input_size = input_size
        self._fm_k = fm_k

    def flow(self, input_tuple):
        self._input_, self._input_gradient = input_, input_gradient = input_tuple
        input_size, fm_k = self._input_size, self._fm_k
        assert input_.shape == input_gradient.shape == (input_size, fm_k)
        self._output_ = output_ = np.zeros(fm_k)
        self._output_gradient = output_gradient = np.zeros(fm_k)
        return ( output_, output_gradient )

    def forward(self):
        input_, output_ = self._input_, self._output_
        input_size, fm_k = self._input_size, self._fm_k
        assert input_.shape == (input_size, fm_k)
        assert output_.shape == (fm_k, )

        output_.fill(0.)
        for i in range(0, input_size-1):
            for j in range(i+1, input_size):
                output_ += np.dot( input_[i], input_[j] )
        return True

    def backward(self):
        input_, input_gradient, output_, output_gradient = \
                self._input_, self._input_gradient, self._output_, self._output_gradient
        input_size, fm_k = self._input_size, self._fm_k
        assert input_.shape == input_gradient.shape == (input_size, fm_k)
        assert output_.shape == output_gradient.shape == (fm_k, )

        gradient_sum = np.zeros(fm_k)
        for i in range(input_size):
            gradient_sum += input_[i]
        for i in range(input_size):
            input_gradient[i] = gradient_sum - input_[i]
            input_gradient[i] *= output_gradient
        return True

class FC_Layer(Layer):
    def __init__(self, name, model, input_width, output_width, optimizer = SGD_Optimizer()):
        Layer.__init__(self, name, model)
        self._input_width = input_width
        self._output_width = output_width
        self._optimizer = SGD_Optimizer()
        self._init_vars()
    
    def flow(self, input_tuple):
        self._input_, self._input_gradient = input_, input_gradient = input_tuple
        assert input_.shape == (self._input_width,)
        assert input_gradient.shape == (self._input_width,)

        self._output_ = output_ = np.zeros( self._output_width )
        self._output_gradient = output_gradient = np.zeros( self._output_width )
        return (output_, output_gradient)

    def _init_vars(self):
        name = self.name
        input_width = self._input_width
        output_width = self._output_width
        vars_ = self._model.vars
        
        {
            SGD_Optimizer : ( lambda : vars_.update( {
                    name + '_w'  : np.random.normal(scale = 0.5, size = (output_width, input_width)),
                    name + '_b'  : np.zeros(output_width) } ) ),
            RMSProp_Optimizer : ( lambda : vars_.update( {
                    name + '_w'  : np.random.normal(size = (output_width, input_width)),
                    name + '_wr' : np.zeros((output_width, input_width)),
                    name + '_b'  : np.zeros(output_width),
                    name + '_br' : np.zeros(output_width) } ) ),
                }[ type(self._optimizer) ] ()
        return True

    def _get_in_out_w_b(self):
        name = self.name
        input_width = self._input_width
        output_width = self._output_width
        input_, output_ = self._input_, self._output_
        vars_ = self._model.vars
        w_ = vars_[name + '_w']
        b_ = vars_[name + '_b']
        assert input_.shape == (input_width,)
        assert output_.shape == (output_width,)
        assert w_.shape == (output_width, input_width)
        assert b_.shape == (output_width,)
        return input_, output_, w_, b_
 
    def forward(self):
        input_, output_, w_, b_ = self._get_in_out_w_b()
        output_[:] = np.dot( w_, input_ ) + b_
        output_[output_ < 0.0] = 0 # ReLU
        return True

    def backward(self):
        input_, output_, w_, b_ = self._get_in_out_w_b()

        input_width, output_width = self._input_width, self._output_width
        input_gradient , output_gradient = self._input_gradient, self._output_gradient
        assert input_gradient.shape == input_.shape == (input_width, )
        assert output_gradient.shape == output_.shape == (output_width, )

        # Relu
        relu_gradient = output_gradient.copy()
        relu_gradient[output_ < 0.] = 0
        # w
        w_gradient = np.dot ( relu_gradient.reshape(output_width, 1) , input_.reshape(1, input_width) )
        # input
        input_gradient[:] = np.dot ( np.transpose(w_) , relu_gradient )
        # b
        b_gradient = relu_gradient

        #update
        update_func = self._optimizer.update
        vars_ = self._model.vars
        assert {
                SGD_Optimizer : (lambda : update_func(w_, w_gradient) and update_func(b_, b_gradient) ),
                RMSProp_Optimizer : (lambda : update_func(w_, vars_[self.name + '_wr'], w_gradient)
                        and update_func(b_, vars_[self.name + '_br'], b_gradient) )
                }[ type(self._optimizer) ]()
        return True

class SGDNN_Softmax(Fid_Eliminate_Model):
    def __init__(self,
            class_num,
            slotslist,
            fid_eliminate=False, fid_expire_ts = 3600*12*7):
        Fid_Eliminate_Model.__init__(self, fid_eliminate, fid_expire_ts)

        self._vector_embedding = Sparse_Vector_Embedding_layer('sve1', self, slotslist, output_type = 'flat', vectorsize = 5,
                optimizer = Ftrl_Optimizer())
        self._nn1 = FC_Layer('nn1', self, 
                input_width = self._vector_embedding.get_output_width(),
                output_width = 10,
                optimizer = RMSProp_Optimizer(),
                )
        self._nn2 = FC_Layer('nn2', self, 
                input_width = 10,
                output_width = 10,
                optimizer = RMSProp_Optimizer(),
                )
        self._nn3 = FC_Layer('nn3', self, 
                input_width = 10,
                output_width = class_num,
                optimizer = RMSProp_Optimizer(),
                )
        self._softmax = Sparse_Softmax_Cross_Entropy_Layer('softmax', self, class_num)

        self.layers = [
                self._vector_embedding,
                self._nn1,
                self._nn2,
                self._nn3,
                self._softmax
                ]

        self.placeholders = {}
        self.tensors = {}
        sve1_output_tuple = self._vector_embedding.flow()
        nn1_output_tuple = self._nn1.flow(sve1_output_tuple)
        nn2_output_tuple = self._nn2.flow(nn1_output_tuple)
        nn3_output_tuple = self._nn3.flow(nn2_output_tuple)
        pred_probs = self._softmax.flow(nn3_output_tuple)
        self.placeholders['pred_probs']  = pred_probs

        #self.load_model_opts.append(self._init_nn_sgd_optimizer)
        #self.load_model_opts.append(self._load_nn_variables)

    def _load_nn_variables(self):
        nn_var_names = [
                'nn1_w', 'nn1_wr','nn1_b', 'nn1_br',
                ]
        for var_name in nn_var_names:
            self.vars[var_name] = np.array(self.vars[var_name], dtype=float)
        return True

    def _forward(self):
        assert 'x' in self.placeholders
        for layer in self.layers:
            assert layer.forward()
        return True

    def _backward(self):
        assert 'y' in self.placeholders
        for layer in self.layers[::-1]:
            assert layer.backward()
        return True

    def predict(self, x):
        assert type(x) == list
        self.placeholders['x'] = x
        assert self._forward()
        pred_probs = self.placeholders['pred_probs']
        max_likelihood_y = self.placeholders['max_likelihood_y']
        return True, pred_probs, max_likelihood_y

    def update(self, x, y):
        assert type(x) == list
        assert type(y) == int
        self.placeholders['x'] = x
        self.placeholders['y'] = y
        assert self._forward()
        assert self._backward()
        pred_probs = self.placeholders['pred_probs']
        max_likelihood_y = self.placeholders['max_likelihood_y']
        cross_entropy_ = self.placeholders['cross_entropy']
        return True, cross_entropy_, pred_probs, max_likelihood_y

class _UnitTest(unittest.TestCase):
    def test_train(self):
        model = SGDNN_Softmax(3, [0])
        for i in range(100000):
            x = []
            xi = random.randint(1,3)
            x.append(xi)
            y = xi % 2
            probs = model.predict(x)
            print model.update(x, y)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
