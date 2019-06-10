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
