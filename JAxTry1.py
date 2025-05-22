from jax import random
from jax import grad, jit, vmap, pmap
import numpy as np
import jax
import jax.numpy as jnp

#x = jnp.linspace(0,10, 1000)

seed = 0
key = random.PRNGKey(seed)

#x = random.normal(key, (3000, 3000))
#dot = jnp.dot(x, x.T).block_until_ready()

x = 1.
y = 1.

f = lambda x,y: x**2 + x + 4 + y**2

dfdx = grad(f)
d2fdx = grad(dfdx)
d3fdx = grad(d2fdx)

#print(f(x,y), dfdx(x,y), d2fdx(x,y), d3fdx(x,y))

@jit
def f(x,y):
    print("Running f()")
    print(f" x = {x}")
    print(f" y = {y}")
    result = jnp.dot(x+1, y+1)
    return result

x = np.random.rand(3, 4)
y = np.random.randn(4)

print(f(x,y))





