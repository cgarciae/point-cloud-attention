# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of multiheaded attention and self-attention layers."""


import tensorflow as tf


class DenseMultiHead(tf.keras.layers.Layer):
    """A Dense Layer using 3D kernel with tf.einsum implementation.
  Attributes:
    num_attention_heads: An integer, number of attention heads for each
      multihead attention layer.
    size_per_head: An integer, hidden size per attention head.
    hidden_size: An integer, dimension of the hidden layer.
    kernel_initializer: An initializer for the kernel weight.
    bias_initializer: An initializer for the bias.
    activation: An activation function to use. If nothing is specified, no
      activation is applied.
    use_bias: A bool, whether the layer uses a bias.
  """

    def __init__(
        self,
        num_attention_heads=12,
        size_per_head=72,
        kernel_initializer=None,
        bias_initializer="zeros",
        activation=None,
        use_bias=True,
        **kwargs
    ):
        """Inits DenseMultiHead."""
        super(DenseMultiHead, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.hidden_size = num_attention_heads * size_per_head
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias

    @property
    def kernel_shape(self):
        return [self.last_dim, self.num_attention_heads, self.size_per_head]

    @property
    def bias_shape(self):
        return [self.num_attention_heads, self.size_per_head]

    def build(self, input_shape):
        """Implements build() for the layer."""
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `DenseMultiHead` layer with non-floating "
                "point (and non-complex) dtype %s" % (dtype,)
            )
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the inputs to `DenseMultiHead` "
                "should be defined. Found `None`."
            )
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: self.last_dim}
        )

        kernel_shape = self.kernel_shape
        bias_shape = self.bias_shape

        self.kernel = self.add_weight(
            "kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs):
        """Implements ``call()`` for DenseMultiHead.
    Args:
      inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
        when output_projection is False, otherwise a float tensor of shape
        [batch_size, sequence_length, num_heads, dim_per_head].
    Returns:
      The projected tensor with shape [batch_size, sequence_length, num_heads,
        dim_per_head] when output_projection is False, otherwise [batch_size,
        sequence_length, hidden_size].
    """

        kernel = self.kernel
        bias = self.bias

        ret = tf.einsum("abc,cde->abde", inputs, kernel)

        if self.use_bias:
            ret += bias

        if self.activation is not None:
            return self.activation(ret)

        return ret


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, size_per_head, num_heads, output_size=None):
        """Initialize Attention.

    Args:
      size_per_head: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      dropout_rate: float, dropout rate inside attention for training.
    """

        super().__init__()
        self.size_per_head = size_per_head
        self.num_heads = num_heads
        self.output_size = output_size

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.

        output_size = (
            self.output_size if self.output_size is not None else input_shape[-1]
        )

        self.query_dense = DenseMultiHead(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="query",
        )
        self.key_dense = DenseMultiHead(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="key",
        )
        self.value_dense = DenseMultiHead(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="value",
        )
        self.projection_kernel = self.add_weight(
            "projection_kernel",
            shape=[self.num_heads, self.size_per_head, output_size],
            initializer="glorot_uniform",
            dtype=self.dtype,
            trainable=True,
        )

        super().build(input_shape)

    def get_config(self):
        return {
            "size_per_head": self.size_per_head,
            "num_heads": self.num_heads,
            "output_size": self.output_size,
        }

    def call(
        self, query, key, value=None, mask=None, training=None,
    ):
        """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size]

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
        # Linearly project the query, key and value using different learned
        # projections. Splitting heads is automatically done during the linear
        # projections --> [batch_size, length, num_heads, dim_per_head].

        if value is None:
            value = key

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        projection = self.projection_kernel

        attention_output = multi_head_attention(
            query, key, value, projection, mask=mask
        )

        return attention_output


class MultiHeadSelfAttention(MultiHeadAttention):
    def call(self, inputs, **kwargs):
        return super().call(inputs, inputs, **kwargs)


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, size_per_head, num_heads, activation="relu"):
        super().__init__()
        self.size_per_head = size_per_head
        self.num_heads = num_heads
        self.activation = activation

    def build(self, input_shape):

        output_size = input_shape[-1]

        self.mha = MultiHeadAttention(
            self.size_per_head, self.num_heads, output_size=output_size
        )
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.layers.Dense(output_size, activation=self.activation)
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        super().build(input_shape)

    def call(self, x, y, mask=None, training=None):

        x = x + self.mha(x, y, mask=mask, training=training)
        x = self.layer_norm1(x)
        x = x + self.ffn(x)
        x = self.layer_norm2(x)

        return x


class MultiHeadSelfAttentionBlock(MultiHeadAttentionBlock):
    def call(self, x, **kwargs):
        return super().call(x, x, **kwargs)


class InducedMultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self, size_per_head, num_heads, num_inducing_points, activation="relu"
    ):
        super().__init__()

        self.size_per_head = size_per_head
        self.num_heads = num_heads
        self.num_inducing_points = num_inducing_points
        self.activation = activation

    def build(self, input_shape):

        output_size = input_shape[-1]

        self.mab1 = MultiHeadAttentionBlock(
            self.size_per_head, self.num_heads, activation=self.activation
        )
        self.mab2 = MultiHeadAttentionBlock(
            self.size_per_head, self.num_heads, activation=self.activation
        )

        self.inducing_kernel = self.add_weight(
            "inducing_kernel",
            shape=[1, self.num_inducing_points, output_size],
            initializer="glorot_uniform",
            dtype=self.dtype,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x, y, mask=None):

        batch_size = tf.shape(x)[0]
        inducing_points = tf.tile(self.inducing_kernel, [batch_size, 1, 1])

        h = self.mab1(inducing_points, y)
        output = self.mab2(x, h)

        return output


class InducedMultiHeadSelfAttentionBlock(InducedMultiHeadAttentionBlock):
    def __call__(self, x, **kwargs):
        return super().__call__(x, x, **kwargs)


class MultiHeadAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size, size_per_head, num_heads, activation="relu"):
        super().__init__()

        self.size_per_head = size_per_head
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.activation = activation

    def build(self, input_shape):

        output_size = input_shape[-1]

        self.mab = MultiHeadAttentionBlock(
            self.size_per_head, self.num_heads, activation=self.activation
        )

        if self.pool_size > 1:
            self.sab = MultiHeadSelfAttentionBlock(
                self.size_per_head, self.num_heads, activation=self.activation
            )
        else:
            self.sab = None

        self.inducing_kernel = self.add_weight(
            "inducing_kernel",
            shape=[self.pool_size, output_size],
            initializer="glorot_uniform",
            dtype=self.dtype,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x, mask=None):

        batch_size = tf.shape(x)[0]

        # add and tile batch dimension
        inducing_points = tf.tile(self.inducing_kernel[None], [batch_size, 1, 1])

        x = self.mab(inducing_points, x)

        if self.sab is not None:
            x = self.sab(x)

        return x


################################################################
# functions
################################################################


def multi_head_attention(query, key, value, projection, mask=None):

    depth = tf.cast(tf.shape(query)[-1], tf.float32)

    query /= tf.sqrt(depth)

    # Calculate dot product attention
    logits = tf.einsum("BTNH,BFNH->BNFT", key, query)

    # apply mask
    if mask is not None:
        logits += mask

    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    attention = tf.nn.softmax(logits, name="attention_weights")

    concated_output = tf.einsum("BNFT,BTNH->BFNH", attention, value)

    # Run the outputs through another linear projection layer. Recombining heads
    # is automatically done --> [batch_size, length, hidden_size]
    attention_output = tf.einsum("BFNH,NHD->BFD", concated_output, projection)

    return attention_output
