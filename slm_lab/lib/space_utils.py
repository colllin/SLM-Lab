import numpy as np
import ctypes
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict, Tuple
from gym.spaces.utils import flatdim


_NP_TO_CT = {
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_char,
    np.bool: ctypes.c_bool,
}


def create_shmem_space(multiproc_context, space):
    if np.any([isinstance(space, t) for t in [Box, Discrete, MultiBinary, MultiDiscrete]]):
        return multiproc_context.Array(_NP_TO_CT[space.dtype], flatdim(space))
    elif isinstance(space, Dict):
        class TempStruct(ctypes.Structure):
            _fields_ = [(key, create_shmem_observation(multiproc_context, subspace)) for key, subspace in space.spaces.items()]
        return TempStruct()
    elif isinstance(space, Tuple):
        class TempStruct(ctypes.Structure):
            _fields_ = [(i, create_shmem_observation(multiproc_context, subspace)) for i, subspace in enumerate(space.spaces)]
        return TempStruct()
    else:
        raise NotImplementedError

    
def write_shmem_space(shmem_space, space, val):
    if np.any([isinstance(space, t) for t in [Box, Discrete, MultiBinary, MultiDiscrete]]):
        dst_np = np.frombuffer(shmem_space.get_obj(), dtype=space.dtype)
        src_flat = flatten(space, val)
        np.copyto(dst_np, src_flat)
    elif isinstance(space, Dict):
        for key, subval in val.items():
            write_shmem_space(getattr(shmem_space, key), subval)
    elif isinstance(space, Tuple):
        for key, subval in enumerate(val):
            write_shmem_space(getattr(shmem_space, key), subval)
    else:
        raise NotImplementedError


def read_shmem_space(shmem_space, space):
    if np.any([isinstance(space, t) for t in [Box, Discrete, MultiBinary, MultiDiscrete]]):
        return unflatten(space, np.frombuffer(shmem_space.get_obj()))
    elif isinstance(space, Dict):
        return {
            key: read_shmem_space(getattr(shmem_space, key), subspace)
        for key, subspace in space.spaces.items()}
    elif isinstance(space, Tuple):
        return (
            read_shmem_space(getattr(shmem_space, key), subspace)
        for key, subspace in enumerate(space.spaces))
    else:
        raise NotImplementedError


# Modified to use space.dtype from https://github.com/openai/gym/blob/master/gym/spaces/utils.py
def flatten(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, MultiBinary):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, Discrete):
        onehot = np.zeros(space.n, dtype=space.dtype)
        onehot[x] = 1.0
        return onehot
    elif isinstance(space, Tuple):
        return np.concatenate([flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, Dict):
        return np.concatenate([flatten(space.spaces[key], item) for key, item in x.items()])
    else:
        raise NotImplementedError


# Modified to use space.dtype from https://github.com/openai/gym/blob/master/gym/spaces/utils.py
def unflatten(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=space.dtype).reshape(space.shape)
    elif isinstance(space, Discrete):
        return int(np.nonzero(x)[0][0])
    elif isinstance(space, MultiBinary):
        return np.asarray(x, dtype=space.dtype).reshape(space.shape)
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x, dtype=space.dtype).reshape(space.shape)
    elif isinstance(space, Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [unflatten(s, flattened) 
                            for flattened, s in zip(list_flattened, space.spaces)]
        return tuple(list_unflattened)
    elif isinstance(space, Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [(key, unflatten(s, flattened)) 
                            for flattened, (key, s) in zip(list_flattened, space.spaces.items())]
        return dict(list_unflattened)
    else:
        raise NotImplementedError
