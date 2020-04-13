# Wrappers for parallel vector environments.
# Adapted from OpenAI Baselines (MIT) https://github.com/openai/baselines/tree/master/baselines/common/vec_env
import gym
import numpy as np
from .eventful import Eventful
from slm_lab.lib import logger



def tile_images(img_nhwc):
    '''
    Tile N images into a rectangular grid for rendering

    @param img_nhwc list or array of images, with shape (batch, h, w, c)
    @returns bigim_HWc ndarray with shape (h',w',c)
    '''
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


@Eventful('EPISODE_STARTED', 'STEP_BATCH', 'EPISODE_DONE')
class EpisodePool(object):    
    viewer = None
    
    def __init__(self, agent, envs):
        '''
        agent: must implement `act` method which receives an observation. Recommended to bind to the events of this class for more complex agents and for training.
        '''
        self.agent = agent
        self.envs = envs
        self.obs1_batch = self.action_batch = self.reward_batch = self.info_batch = self.obs2_batch = [None for env in self.envs]
        self.done_batch = [True for env in self.envs]
               
    # Just create a new EpisodePool with the same envs...
    # def reset(self):
    #     '''
    #     Reset all the environments and return an array of observations, or a dict of observation arrays.
    # 
    #     If step_async is still doing work, that work will be cancelled and step_wait() should not be called until step_async() is invoked again.
    #     '''
    #     pass

    def _init_episodes(self):
        for i, env in enumerate(self.envs):
            if self.done_batch[i]:
                self.obs1_batch[i] = env.reset()
                self.done_batch[i] = False
                self.emit('EPISODE_STARTED', {'env_id': i})
            
    def _teardown_episodes(self):
        for i, env in enumerate(self.envs):
            if self.done_batch[i]:
                self.emit('EPISODE_DONE', {'env_id': i})

    def step(self):
        self.obs1_batch = self.obs2_batch
        self._init_episodes()
        self.action_batch = self.agent.act(self.obs1_batch)
        if hasattr(env, 'step_async'):
            for env, action in zip(self.envs, self.action_batch):
                env.step_async(action)
            self.obs2_batch, self.reward_batch, self.done_batch, self.info_batch = zip([env.step_wait() for env in self.envs])
        else:
            self.obs2_batch, self.reward_batch, self.done_batch, self.info_batch = zip([env.step(action) for env, action in zip(self.envs, self.action_batch)])
        returns = self.obs2_batch, self.reward_batch, self.done_batch, self.info_batch
        self.emit('STEP', (self.obs1_batch, self.action_batch) + returns)
        self._teardown_episodes()
        return returns
        
    def render(self, mode='human'):
        imgs = [env.render(mode='rgb_array') for env in self.envs]
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# We decide to do everything using EpisodePool for simplicity, and to enforce that
# agents always take a "batch" of observations and return a "batch" of actions.
# --------------
# @Eventful('STARTED', 'STEP', 'DONE')
# class Episode:
#     def __init__(self, agent, env):
#         self.agent = agent
#         self.env = env
#         self.done = False
#         self.canceled = False
#         self.o1, self.action, self.o2, self.reward, self.info = None, None, None, None, None
#         self.cumulative_steps = 0
#         self.cumulative_reward = 0
#         def accumulate((o1, act, o2, rew, done, info)):
#             self.cumulative_steps += 1
#             self.cumulative_reward += rew
#         self.subscribe('STEP', accumulate)
#         
#     def _start(self):
#         self.o1 = self.env.reset()
#         self.emit('STARTED', self.o1)
# 
#     def step(self):
#         """Run one step of the episode."""
#         assert not self.done
#         self.o1 = self.o2
#         if not self.o1:
#             self._start()
#         self.action = self.agent.act([self.o1])
#         assert len(self.action) == 1, 'should return 1 action for the batch of 1 environment.
#         self.o2, self.reward, self.done, self.info = self.env.step(self.action[0])
#         self.cumulative_reward += self.reward
#         # self.agent.update(obs, action, reward, next_obs, done)
#         self.emit('STEP', (self.o1, self.action, self.o2, self.reward, self.done, self.info))
#         
#     def run(self):
#         """Steps the episode to completion. Useful for evaluation and deployment."""
#         while not self.done and not self.canceled:
#             self.step()
#             
#     def cancel(self):
#         self.canceled = True


