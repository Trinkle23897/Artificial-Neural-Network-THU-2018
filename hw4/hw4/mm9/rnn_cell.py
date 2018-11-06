"""Module implementing RNN Cells.
This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _like_rnncell(cell):
    """Checks that a given object is an RNNCell by using duck typing."""
    conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
                                hasattr(cell, "zero_state"), callable(cell)]
    return all(conditions)


def _concat(prefix, suffix, static=False):
    """Concat that enables int, Tensor, or TensorShape values.
    This function takes a size specification, which can be an integer, a
    TensorShape, or a Tensor, and converts it into a concatenated Tensor
    (if static = False) or a list of integers (if static = True).
    Args:
        prefix: The prefix; usually the batch size (and/or time step size).
            (TensorShape, int, or Tensor.)
        suffix: TensorShape, int, or Tensor.
        static: If `True`, return a python list with possibly unknown dimensions.
            Otherwise return a `Tensor`.
    Returns:
        shape: the concatenation of prefix and suffix.
    Raises:
        ValueError: if `suffix` is not a scalar or vector (or TensorShape).
        ValueError: if prefix or suffix was `None` and asked for dynamic
            Tensors out.
    """
    if isinstance(prefix, ops.Tensor):
        p = prefix
        p_static = tensor_util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = array_ops.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError("prefix tensor must be either a scalar or vector, "
                                             "but saw tensor: %s" % p)
    else:
        p = tensor_shape.as_shape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
                 if p.is_fully_defined() else None)
    if isinstance(suffix, ops.Tensor):
        s = suffix
        s_static = tensor_util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = array_ops.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError("suffix tensor must be either a scalar or vector, "
                                             "but saw tensor: %s" % s)
    else:
        s = tensor_shape.as_shape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
                 if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s"
                                             % (prefix, suffix))
        shape = array_ops.concat((p, s), 0)
    return shape


def _linear(args,
                        output_size,
                        bias,
                        bias_initializer=None,
                        kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias
            (default is all zeros).
        kernel_initializer: starting value to initialize the weight.
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                    _BIAS_VARIABLE_NAME, [output_size],
                    dtype=dtype,
                    initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)


def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""
    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = _concat(batch_size, s)
        c_static = _concat(batch_size, s, static=True)
        size = array_ops.zeros(c, dtype=dtype)
        size.set_shape(c_static)
        return size
    return nest.map_structure(get_state_shape, state_size)


class RNNCell(base_layer.Layer):
    """Abstract object representing an RNN cell.
    Every `RNNCell` must have the properties below and implement `call` with
    the signature `(output, next_state) = call(input, state)`.    The optional
    third input argument, `scope`, is allowed for backwards compatibility
    purposes; but should be left off for new subclasses.
    This definition of cell differs from the definition used in the literature.
    In the literature, 'cell' refers to an object with a single scalar output.
    This definition refers to a horizontal array of such units.
    An RNN cell, in the most abstract setting, is anything that has
    a state and performs some operation that takes a matrix of inputs.
    This operation results in an output matrix with `self.output_size` columns.
    If `self.state_size` is an integer, this operation also results in a new
    state matrix with `self.state_size` columns.    If `self.state_size` is a
    (possibly nested tuple of) TensorShape object(s), then it should return a
    matching structure of Tensors having shape `[batch_size].concatenate(s)`
    for each `s` in `self.batch_size`.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`.
            state: if `self.state_size` is an integer, this should be a `2-D Tensor`
                with shape `[batch_size x self.state_size]`.    Otherwise, if
                `self.state_size` is a tuple of integers, this should be a tuple
                with shapes `[batch_size x s] for s in self.state_size`.
            scope: VariableScope for the created subgraph; defaults to class name.
        Returns:
            A pair containing:
            - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
            - New state: Either a single `2-D` tensor, or a tuple of tensors matching
                the arity and shapes of `state`.
        """
        if scope is not None:
            with vs.variable_scope(scope,
                                                         custom_getter=self._rnn_get_variable) as scope:
                return super(RNNCell, self).__call__(inputs, state, scope=scope)
        else:
            with vs.variable_scope(vs.get_variable_scope(),
                                                         custom_getter=self._rnn_get_variable):
                return super(RNNCell, self).__call__(inputs, state)

    def _rnn_get_variable(self, getter, *args, **kwargs):
        variable = getter(*args, **kwargs)
        trainable = (variable in tf_variables.trainable_variables() or
                                 (isinstance(variable, tf_variables.PartitionedVariable) and
                                    list(variable)[0] in tf_variables.trainable_variables()))
        if trainable and variable not in self._trainable_weights:
            self._trainable_weights.append(variable)
        elif not trainable and variable not in self._non_trainable_weights:
            self._non_trainable_weights.append(variable)
        return variable

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def build(self, _):
        # This tells the parent Layer object that it's OK to call
        # self.add_variable() inside the call() method.
        pass

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.
        Returns:
            If `state_size` is an int or TensorShape, then the return value is a
            `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
            If `state_size` is a nested list or tuple, then the return value is
            a nested list or tuple (of the same structure) of `2-D` tensors with
            the shapes `[batch_size x s]` for each s in `state_size`.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            state_size = self.state_size
            return _zero_state_tensors(state_size, batch_size, dtype)


class BasicRNNCell(RNNCell):
    """The most basic RNN cell.
    Args:
        num_units: int, The number of units in the LSTM cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
    """

    def __init__(self, num_units, activation=None, reuse=None):
        super(BasicRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """basic RNN: output = new_state = activation(W * input + U * state + B)."""
        output = self._activation(_linear([inputs, state], self._num_units, True))
        return output, output


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self,
                             num_units,
                             activation=None,
                             reuse=None,
                             kernel_initializer=None,
                             bias_initializer=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope("gates"):    # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = math_ops.sigmoid(
                    _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                                    self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            #todo: calculate c and new_h according to GRU
            # Ref to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py
            c = self._activation(_linear([inputs, r * state], self._num_units, False))
        new_h = u * state + (1 - u) * c
        return new_h, new_h


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(c, h)`, in that order.
    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                                            (str(c.dtype), str(h.dtype)))
        return c.dtype


class BasicLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0,
                             state_is_tuple=True, activation=None, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (see above).
            state_is_tuple: If True, accepted and returned states are 2-tuples of
                the `c_state` and `m_state`.    If False, they are concatenated
                along the column axis.    The latter behavior will soon be deprecated.
            activation: Activation function of the inner states.    Default: `tanh`.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.    If not `True`, and the existing scope already has
                the given variables, an error is raised.
        """
        super(BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                                     "deprecated.    Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                        if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM)."""
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        #todo: calculate new_c and new_h according to LSTM
        # Ref to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py
        gate_input = _linear([inputs, h], 4 * self._num_units, bias=True)
        i, f, o, c_ = array_ops.split(value=gate_input, num_or_size_splits=4, axis=1)
        new_c = sigmoid(f + self._forget_bias) * c + sigmoid(i) * self._activation(c_)
        new_h = sigmoid(o) * self._activation(new_c)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


class MultiRNNCell(RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells, state_is_tuple=True):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
            cells: list of RNNCells that will be composed in this order.
            state_is_tuple: If True, accepted and returned states are n-tuples, where
                `n = len(cells)`.    If False, the states are all
                concatenated along the column axis.    This latter behavior will soon be
                deprecated.
        Raises:
            ValueError: if cells is empty (not allowed), or at least one of the cells
                returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(MultiRNNCell, self).__init__()
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        if not nest.is_sequence(cells):
            raise TypeError(
                    "cells must be a list or tuple, but saw: %s." % cells)

        self._cells = cells
        self._state_is_tuple = state_is_tuple
        if not state_is_tuple:
            if any(nest.is_sequence(c.state_size) for c in self._cells):
                raise ValueError("Some cells return tuples of states, but the flag "
                                                 "state_is_tuple is not set.    State sizes are: %s"
                                                 % str([c.state_size for c in self._cells]))

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else fancy
                return super(MultiRNNCell, self).zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError(
                                "Expected state to be a tuple of length %d, but received: %s" %
                                (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                                                                            [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)

        new_states = (tuple(new_states) if self._state_is_tuple else
                                    array_ops.concat(new_states, 1))

        return cur_inp, new_states


