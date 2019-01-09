import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------------------------------------------------------------
# Commonly used layers and operations based on ethereon's implementation
# https://github.com/ethereon/caffe-tensorflow
# Slight modifications may apply. FCRN-specific operations have also been appended.
# ----------------------------------------------------------------------------------
# Thanks to *Helisa Dhamo* for the model conversion and integration into TensorFlow.
# ----------------------------------------------------------------------------------

DEFAULT_PADDING = 'SAME'


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))

        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, batch, keep_prob, is_training, trainable = True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        self.batch_size = batch
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.setup()


    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iter(data_dict[op_name].items()):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))

                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_layer_output(self, name):
        return self.layers[name]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, dtype = 'float32', trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input_data,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):

        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input_data.get_shape()[-1]

        if (padding == 'SAME'):
            input_data = tf.pad(input_data, [[0, 0], [(k_h - 1)//2, (k_h - 1)//2], [(k_w - 1)//2, (k_w - 1)//2], [0, 0]], "CONSTANT")

        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding='VALID')

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])

            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input_data, kernel)
            else:
                # Split the input into groups and then convolve each of them independently

                input_groups = tf.split(3, group, input_data)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)

            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)

            return output

    @layer
    def relu(self, input_data, name):
        return tf.nn.relu(input_data, name=name)

    @layer
    def max_pool(self, input_data, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input_data,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input_data, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input_data,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input_data, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input_data,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input_data, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input_data.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input_data, [-1, dim])
            else:
                feed_in, dim = (input_data, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input_data, name):
        input_shape = map(lambda v: v.value, input_data.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input_data = tf.squeeze(input_data, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input_data, name)

    @layer
    def batch_normalization(self, input_data, name, scale_offset=True, relu=False):

        with tf.variable_scope(name) as scope:
            shape = [input_data.get_shape()[-1]]
            pop_mean = tf.get_variable("mean", shape, initializer = tf.constant_initializer(0.0), trainable=False)
            pop_var = tf.get_variable("variance", shape, initializer = tf.constant_initializer(1.0), trainable=False)
            epsilon = 1e-4
            decay = 0.999
            if scale_offset:
                scale = tf.get_variable("scale", shape, initializer = tf.constant_initializer(1.0))
                offset = tf.get_variable("offset", shape, initializer = tf.constant_initializer(0.0))
            else:
                scale, offset = (None, None)
            if self.is_training:
                batch_mean, batch_var = tf.nn.moments(input_data, [0, 1, 2])

                train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    output = tf.nn.batch_normalization(input_data,
                    batch_mean, batch_var, offset, scale, epsilon, name = name)
            else:
                output = tf.nn.batch_normalization(input_data,
                pop_mean, pop_var, offset, scale, epsilon, name = name)

            if relu:
                output = tf.nn.relu(output)

            return output

    @layer
    def dropout(self, input_data, keep_prob, name):
        return tf.nn.dropout(input_data, keep_prob, name=name)


    def unpool_as_conv(self, size, input_data, id, stride = 1, ReLU = False, BN = True):

		# Model upconvolutions (unpooling + convolution) as interleaving feature
		# maps of four convolutions (A,B,C,D). Building block for up-projections.


        # Convolution A (3x3)
        # --------------------------------------------------
        layerName = "layer%s_ConvA" % (id)
        self.feed(input_data)
        self.conv( 3, 3, size[3], stride, stride, name = layerName, padding = 'SAME', relu = False)
        outputA = self.get_output()

        # Convolution B (2x3)
        # --------------------------------------------------
        layerName = "layer%s_ConvB" % (id)
        padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
        self.feed(padded_input_B)
        self.conv(2, 3, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
        outputB = self.get_output()

        # Convolution C (3x2)
        # --------------------------------------------------
        layerName = "layer%s_ConvC" % (id)
        padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
        self.feed(padded_input_C)
        self.conv(3, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
        outputC = self.get_output()

        # Convolution D (2x2)
        # --------------------------------------------------
        layerName = "layer%s_ConvD" % (id)
        padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
        self.feed(padded_input_D)
        self.conv(2, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
        outputD = self.get_output()

        # Interleaving elements of the four feature maps
        # --------------------------------------------------
        left = interleave([outputA, outputB], axis=1)  # columns
        right = interleave([outputC, outputD], axis=1)  # columns
        Y = interleave([left, right], axis=2) # rows

        if BN:
            layerName = "layer%s_BN" % (id)
            self.feed(Y)
            self.batch_normalization(name = layerName, scale_offset = True, relu = False)
            Y = self.get_output()

        if ReLU:
            Y = tf.nn.relu(Y, name = layerName)

        return Y


    def up_project(self, size, id, stride = 1, BN = True):

        # Create residual upsampling layer (UpProjection)

        input_data = self.get_output()

        # Branch 1
        id_br1 = "%s_br1" % (id)

        # Interleaving Convs of 1st branch
        out = self.unpool_as_conv(size, input_data, id_br1, stride, ReLU=True, BN=True)

        # Convolution following the upProjection on the 1st branch
        layerName = "layer%s_Conv" % (id)
        self.feed(out)
        self.conv(size[0], size[1], size[3], stride, stride, name = layerName, relu = False)

        if BN:
            layerName = "layer%s_BN" % (id)
            self.batch_normalization(name = layerName, scale_offset=True, relu = False)

        # Output of 1st branch
        branch1_output = self.get_output()


        # Branch 2
        id_br2 = "%s_br2" % (id)
        # Interleaving convolutions and output of 2nd branch
        branch2_output = self.unpool_as_conv(size, input_data, id_br2, stride, ReLU=False)


        # sum branches
        layerName = "layer%s_Sum" % (id)
        output = tf.add_n([branch1_output, branch2_output], name = layerName)
        # ReLU
        layerName = "layer%s_ReLU" % (id)
        output = tf.nn.relu(output, name=layerName)

        self.feed(output)
        return self



#######################################################################

class ResNet50UpProj(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
             .batch_normalization(relu=True, name='bn_conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(name='bn2a_branch1'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(relu=True, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(relu=True, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(relu=True, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(relu=True, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(relu=True, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(relu=True, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(name='bn3a_branch1'))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(relu=True, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(relu=True, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
             .batch_normalization(relu=True, name='bn3b_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
             .batch_normalization(relu=True, name='bn3b_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
             .batch_normalization(name='bn3b_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b_branch2c')
             .add(name='res3b')
             .relu(name='res3b_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
             .batch_normalization(relu=True, name='bn3c_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
             .batch_normalization(relu=True, name='bn3c_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
             .batch_normalization(name='bn3c_branch2c'))

        (self.feed('res3b_relu',
                   'bn3c_branch2c')
             .add(name='res3c')
             .relu(name='res3c_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
             .batch_normalization(relu=True, name='bn3d_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
             .batch_normalization(relu=True, name='bn3d_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
             .batch_normalization(name='bn3d_branch2c'))

        (self.feed('res3c_relu',
                   'bn3d_branch2c')
             .add(name='res3d')
             .relu(name='res3d_relu')
             .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(name='bn4a_branch1'))

        (self.feed('res3d_relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(relu=True, name='bn4a_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(relu=True, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(name='bn4a_branch2c'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
             .batch_normalization(relu=True, name='bn4b_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
             .batch_normalization(relu=True, name='bn4b_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
             .batch_normalization(name='bn4b_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b_branch2c')
             .add(name='res4b')
             .relu(name='res4b_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
             .batch_normalization(relu=True, name='bn4c_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
             .batch_normalization(relu=True, name='bn4c_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
             .batch_normalization(name='bn4c_branch2c'))

        (self.feed('res4b_relu',
                   'bn4c_branch2c')
             .add(name='res4c')
             .relu(name='res4c_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
             .batch_normalization(relu=True, name='bn4d_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
             .batch_normalization(relu=True, name='bn4d_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
             .batch_normalization(name='bn4d_branch2c'))

        (self.feed('res4c_relu',
                   'bn4d_branch2c')
             .add(name='res4d')
             .relu(name='res4d_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
             .batch_normalization(relu=True, name='bn4e_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
             .batch_normalization(relu=True, name='bn4e_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
             .batch_normalization(name='bn4e_branch2c'))

        (self.feed('res4d_relu',
                   'bn4e_branch2c')
             .add(name='res4e')
             .relu(name='res4e_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
             .batch_normalization(relu=True, name='bn4f_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
             .batch_normalization(relu=True, name='bn4f_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
             .batch_normalization(name='bn4f_branch2c'))

        (self.feed('res4e_relu',
                   'bn4f_branch2c')
             .add(name='res4f')
             .relu(name='res4f_relu')
             .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(name='bn5a_branch1'))

        (self.feed('res4f_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(relu=True, name='bn5a_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(relu=True, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(name='bn5a_branch2c'))

        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(relu=True, name='bn5b_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(relu=True, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(relu=True, name='bn5c_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(relu=True, name='bn5c_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='layer1')
             .batch_normalization(relu=False, name='layer1_BN')
             .up_project([3, 3, 1024, 512], id = '2x', stride = 1, BN=True)
             .up_project([3, 3, 512, 256], id = '4x', stride = 1, BN=True)
             .up_project([3, 3, 256, 128], id = '8x', stride = 1, BN=True)
             .up_project([3, 3, 128, 64], id = '16x', stride = 1, BN=True)
             .dropout(name = 'drop', keep_prob = 1.)
             .conv(3, 3, 1, 1, 1, name = 'ConvPred'))




import os
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import time

#import models

#model_data_path= "NYU_ResNet-UpProj.npy"
model_data_path= "NYU_FCRN.ckpt"
#image_path= "Capture2.JPG"
#image_path="Class.JPG"

#Check local device
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())



def construct_net(sess, input_height, input_width, channels, batch_size):
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, input_height, input_width, channels))

    # Construct the network
    net = ResNet50UpProj({'data': input_node}, batch_size, 1, False)


    # Load the converted parameters
    print('Loading the model')

    # Use to load from ckpt file
    saver = tf.train.Saver()
    saver.restore(sess, model_data_path)

    #Use to load from npy file
    #net.load(model_data_path, sess)
    print('Done')

    return input_node, net



def preprocessing(frame, height, width):

    # Read image
    img = Image.fromarray(frame)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    return img

def pred(sess,net,input_node,img):

    # Evalute the network for the given image
    predict = sess.run(net.get_output(), feed_dict={input_node: img})

    #predict_uint8 = cv2.convertScaleAbs(predict[0])
    #cv2.imshow('pred', predict_uint8)
    #predict_colormap = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    pred_median = cv2.medianBlur(predict[0], 3)
    pred_median_scaled_uint8 = cv2.convertScaleAbs(pred_median * (255 / np.max(pred_median)))
    pred_jet = cv2.applyColorMap(255 - pred_median_scaled_uint8, cv2.COLORMAP_JET)
    #pred_jet_resized = cv2.resize(pred_jet, (304, 228), interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow('pred (scaled, median)', pred_median_scaled_uint8)
    cv2.imshow('pred_jet', pred_jet)

    # Plot result
    '''
    fig = plt.figure()
    ii = plt.imshow(predict[0,:,:,0], interpolation='nearest')
    fig.colorbar(ii)
    plt.show()
    cv2.imshow('my webcam', pred)
'''
    return predict





#video_capture = cv2.VideoCapture('drone_Trim.mp4')



def depth_pred(video_capture):

    # Definition of the session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #input_height= 228
    #input_width= 304
    input_height= 480
    input_width= 640
    channels = 3
    batch_size = 1

    input_node, net = construct_net(sess, input_height, input_width, channels, batch_size)


    n_frames = 0
    seconds = 0.0
    global pred
    global fps
    fps= 0.0
    global avg_fps
    avg_fps= []

    while True:
        # Start time
        #start = time.time()
        # Capture frame-by-frame
        _, frame = video_capture.read()
        #n_frames = n_frames + 1

        #preprocessed_image = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)

        # Preprocess the input image
        preprocessed_image = preprocessing(frame,input_height,input_width)


        # Compute the depth predictions on the input image
        #prediction = []
        prediction = pred(sess,net,input_node,preprocessed_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # End time
        #end = time.time()

        # Time elapsed
        #seconds = (end - start)
        # Calculate frames per second
        #note: change fps
        #fps  =  (1/seconds)
        #avg_fps.append(fps)


    #avg_fps= np.mean(avg_fps)
    #video_capture.release()
    cv2.destroyAllWindows()
    print('Done!!')
    #sess.close()



    # Predict the image
    #pred = predict(model_data_path, image_path)

    #os._exit(0)


'''
def main():

    #input_height= 228
    #input_width= 304
    input_height= 480
    input_width= 640
    channels = 3
    batch_size = 1


    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, input_height, input_width, channels))

    # Construct the network
    net = ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    # Definition of the session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Load the converted parameters
    print('Loading the model')

    # Use to load from ckpt file
    saver = tf.train.Saver()
    saver.restore(sess, model_data_path)

    #Use to load from npy file
    #net.load(model_data_path, sess)

    video_capture = cv2.VideoCapture(0)
    n_frames = 0
    seconds = 0.0
    fps = 0.0

    for i in range(3):

        # Start time
        start = time.time()

        # Capture frame-by-frame
        _, frame = video_capture.read()
        n_frames = n_frames + 1

        # Preprocess the input image
        preprocessed_image = preprocessing(frame,input_height,input_width)

        # Compute the depth predictions on the input image
        prediction = []
        prediction = pred(sess,net,input_node,preprocessed_image)
        

    # Predict the image
    #pred = predict(model_data_path, image_path)

    #os._exit(0)

'''
'''
if __name__ == '__main__':
    main()
'''
