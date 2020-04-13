import gym
from .registration import try_register_env, get_env_path
from gym_unity.envs import UnityEnv
from .concurrent_env import ConcurrentEnv



def make_env(env_spec, concurrency=None):
    '''Makes an environment from the spec. `concurrency` determines whether/how the env is run in a sub-process.'''
    if concurrency is None:
        try:
            try_register_env(env_spec['name'])
            return gym.make(
                env_spec['name'], 
                *env_spec.get('args', []), 
                **env_spec.get('kwargs', {})
            )
        except:
            worker_id = int(f'{os.getpid()}{ps.unique_id()}'[-4:])
            return UnityEnv(
                get_env_path(env_spec['name']),
                worker_id=worker_id,
                *env_spec.get('args', []),
                **env_spec.get('kwargs', {})
            )
    return ConcurrentEnv(env_spec, concurrency=concurrency)
