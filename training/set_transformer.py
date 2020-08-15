import jax
import jax.numpy as jnp

import elegy
from elegy.nn.multi_head_attention import MultiHeadAttention
import numpy as np


class MAB(elegy.Module):
    def __init__(self, head_size, num_heads, activation=jax.nn.relu):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.activation = activation

    def call(self, x, y=None, mask=None, training=None):

        output_size = x.shape[-1]

        x = x + MultiHeadAttention(
            self.head_size, self.num_heads, output_size=output_size
        )(x, y, mask=mask, training=training)
        x = elegy.nn.LayerNormalization()(x)
        x = x + elegy.nn.sequential(elegy.nn.Linear(output_size), self.activation)(x)
        x = elegy.nn.LayerNormalization()(x)

        return x


class IMAB(elegy.Module):
    def __init__(self, head_size, num_heads, inducing_points, activation=jax.nn.relu):
        super().__init__()

        self.head_size = head_size
        self.num_heads = num_heads
        self.inducing_points = inducing_points
        self.activation = activation

    def call(self, x, y=None, mask=None):

        batch_size = x.shape[0]
        output_size = x.shape[-1]

        inducing_kernel = elegy.get_parameter(
            "inducing_kernel",
            [1, self.inducing_points, output_size],
            jnp.float32,
            elegy.initializers.TruncatedNormal(stddev=1.0 / np.sqrt(output_size)),
        )

        inducing_points = jnp.tile(inducing_kernel, [batch_size, 1, 1])

        h = MAB(self.head_size, self.num_heads, activation=self.activation)(
            inducing_points, y
        )
        output = MAB(self.head_size, self.num_heads, activation=self.activation)(x, h)

        return output


class PMA(elegy.Module):
    def __init__(self, pool_size, head_size, num_heads, activation=jax.nn.relu):
        super().__init__()

        self.head_size = head_size
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.activation = activation

    def call(self, x, mask=None):
        batch_size = x.shape[0]
        output_size = x.shape[-1]

        inducing_kernel = elegy.get_parameter(
            "inducing_kernel",
            [1, self.pool_size, output_size],
            jnp.float32,
            elegy.initializers.TruncatedNormal(stddev=1.0 / np.sqrt(output_size)),
        )

        # add and tile batch dimension
        inducing_points = jnp.tile(inducing_kernel, [batch_size, 1, 1])

        x = MAB(self.head_size, self.num_heads, activation=self.activation)(
            inducing_points,
            elegy.nn.sequential(elegy.nn.Linear(output_size), self.activation)(x),
        )

        if self.pool_size > 1:
            x = MAB(self.head_size, self.num_heads, activation=self.activation)(x)

        return x
