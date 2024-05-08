# Sampler, for sampling from a gen_fn interactively. 
import jax
import os

def make_sampler():
    k0 = [jax.random.PRNGKey(314159)]
    def inner(gf, *args):
        gf.simulate(jax.random.split(k0[0], 2)[1], args).get_retval()
    return inner    

