import os
import contextlib
import gym
import torch.multiprocessing as mp
from .make_env import make_env
from slm_lab.lib.space_utils import create_shmem_space, read_shmem_space, write_shmem_space


@contextlib.contextmanager
def clear_mpi_env_vars():
    '''
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing Processes.
    '''
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def subproc_worker(pipe, env_spec, shmem_observation):
    '''
    Control a single environment instance using IPC and shared memory. Used by ConcurrentEnv.
    '''
    env = make_env(env_spec, concurrency=None)
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                obs = env.reset()
                write_shmem_space(shmem_observation, env.observation_space, obs)
                pipe.send(None)
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                write_shmem_space(shmem_observation, env.observation_space, obs)
                pipe.send((None, reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError(f'Got unrecognized cmd {cmd}')
    except KeyboardInterrupt:
        logger.exception('ConcurrentEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class ConcurrentEnv(gym.Wrapper):
    '''
    Optimized version of SubprocVecEnv that uses shared memory to communicate observations.
    '''

    def __init__(self, env_spec, context='fork'):
        self.env_spec = env_spec
        with make_env(self.env_spec, concurrency=None) as dummy:
            self.observation_space = dummy.observation_space
        ctx = mp.get_context(context)
#         VecEnv.__init__(self, envs)
        with clear_mpi_env_vars():
            self.shmem_observation = create_shmem_space(ctx, self.observation_space)
            self.parent2child_pipe, child2parent_pipe = ctx.Pipe()
            self.child_proc = ctx.Process(
                target=subproc_worker,
                args=(child2parent_pipe, env_spec, self.shmem_observation)
            )
            self.child_proc.daemon = True
            self.child_proc.start()
            child2parent_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            logger.warning('Called reset() while waiting for the step to complete')
            self.step_wait()
        self.parent2child_pipe.send(('reset', None))
        self.parent2child_pipe.recv()
        obs = read_shmem_space(self.shmem_observation, self.observation_space)
        return obs

    def step_async(self, action):
        '''
        Tell all the environments to start taking a step with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is already pending.
        '''
        self.parent2child_pipe.send(('step', action))

    def step_wait(self):
        '''
        Wait for the step taken with step_async().

        @returns (obs, rews, dones, infos)
         - obs: an array of observations, or a dict of arrays of observations.
         - rews: an array of rewards
         - dones: an array of 'episode done' booleans
         - infos: a sequence of info objects
        '''
        _, rew, done, info = self.parent2child_pipe.recv()
        obs = read_shmem_space(self.shmem_observation, self.observation_space)
        return obs, rew, done, info

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        self.parent2child_pipe.send(('close', None))
        self.parent2child_pipe.recv()
        self.parent2child_pipe.close()
        self.child_proc.join()

    def get_images(self, mode='human'):
        self.parent2child_pipe.send(('render', None))
        return self.parent2child_pipe.recv()
        

