'''
The agent module
Contains graduated components from experiments for building agents and be taught, tested, evaluated on curriculum.
To be designed by human and evolution module, based on the experiment aim (trait) and fitness metrics.
Main SLM components (refer to SLM doc for more):
- primary survival objective
- control policies
- sensors (input) for embodiment
- motors (output) for embodiment
- neural architecture
- memory (with time)
- prioritization mechanism and "emotions"
- strange loop must be created
- social aspect
- high level properties of thinking, e.g. creativity, planning.

Agent components:
- algorithm (with net, policy)
- memory (per body)
'''
from slm_lab.agent import algorithm, memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import torch

AGENT_DATA_NAMES = ['action', 'loss', 'explore_var']
logger = logger.get_logger(__name__)


class Agent:
    '''
    Class for all Agents.
    Standardizes the Agent design to work in Lab.
    Access Envs properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, body, a=None, agent_space=None, global_nets=None):
        self.spec = spec
        self.a = a or 0  # for compatibility with agent_space
        self.agent_spec = spec['agent'][self.a]
        self.name = self.agent_spec['name']
        assert not ps.is_list(global_nets), f'single agent global_nets must be a dict, got {global_nets}'
        if agent_space is None:  # singleton mode
            self.body = body
            body.agent = self
            MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
            self.body.memory = MemoryClass(self.agent_spec['memory'], self.body)
            AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
            self.algorithm = AlgorithmClass(self, global_nets)
        else:
            self.space_init(agent_space, body, global_nets)

        logger.info(util.self_desc(self))

    @lab_api
    def act(self, state):
        '''Standard act method from algorithm.'''
        with torch.no_grad():  # for efficiency, only calc grad in algorithm.train
            action = self.algorithm.act(state)
        return action

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''
        self.body.update(state, action, reward, next_state, done)
        if util.in_eval_lab_modes():  # eval does not update agent for training
            return
        self.body.memory.update(state, action, reward, next_state, done)
        loss = self.algorithm.train()
        if not np.isnan(loss):  # set for log_summary()
            self.body.loss = loss
        explore_var = self.algorithm.update()
        return loss, explore_var

    @lab_api
    def save(self, ckpt=None):
        '''Save agent'''
        if util.in_eval_lab_modes():  # eval does not save new models
            return
        self.algorithm.save(ckpt=ckpt)

    @lab_api
    def close(self):
        '''Close and cleanup agent at the end of a session, e.g. save model'''
        self.save()

    @lab_api
    def space_init(self, agent_space, body_a, global_nets):
        '''Post init override for space env. Note that aeb is already correct from __init__'''
        self.agent_space = agent_space
        self.body_a = body_a
        self.aeb_space = agent_space.aeb_space
        self.nanflat_body_a = util.nanflatten(self.body_a)
        for idx, body in enumerate(self.nanflat_body_a):
            if idx == 0:  # NOTE set default body
                self.body = body
            body.agent = self
            body.nanflat_a_idx = idx
            MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
            body.memory = MemoryClass(self.agent_spec['memory'], body)
        self.body_num = len(self.nanflat_body_a)
        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        self.algorithm = AlgorithmClass(self, global_nets)
        # after algo init, transfer any missing variables from default body
        for idx, body in enumerate(self.nanflat_body_a):
            for k, v in vars(self.body).items():
                if util.gen_isnan(getattr(body, k, None)):
                    setattr(body, k, v)

    @lab_api
    def space_act(self, state_a):
        '''Standard act method from algorithm.'''
        with torch.no_grad():
            action_a = self.algorithm.space_act(state_a)
        return action_a

    @lab_api
    def space_update(self, state_a, action_a, reward_a, next_state_a, done_a):
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''
        for eb, body in util.ndenumerate_nonan(self.body_a):
            body.update(state_a[eb], action_a[eb], reward_a[eb], next_state_a[eb], done_a[eb])
            body.memory.update(state_a[eb], action_a[eb], reward_a[eb], next_state_a[eb], done_a[eb])
        loss_a = self.algorithm.space_train()
        loss_a = util.guard_data_a(self, loss_a, 'loss')
        for eb, body in util.ndenumerate_nonan(self.body_a):
            if not np.isnan(loss_a[eb]):  # set for log_summary()
                body.loss = loss_a[eb]
        explore_var_a = self.algorithm.space_update()
        explore_var_a = util.guard_data_a(self, explore_var_a, 'explore_var')
        # TODO below scheduled for update to be consistent with non-space mode
        for eb, body in util.ndenumerate_nonan(self.body_a):
            if body.env.done:
                body.train_ckpt()
        return loss_a, explore_var_a


class AgentSpace:
    '''
    Subspace of AEBSpace, collection of all agents, with interface to Session logic; same methods as singleton agents.
    Access EnvSpace properties by: AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, aeb_space, global_nets=None):
        self.spec = spec
        self.aeb_space = aeb_space
        aeb_space.agent_space = self
        self.aeb_shape = aeb_space.aeb_shape
        assert not ps.is_dict(global_nets), f'multi agent global_nets must be a list of dicts, got {global_nets}'
        assert ps.is_list(self.spec['agent'])
        self.agents = []
        for a in range(len(self.spec['agent'])):
            body_a = self.aeb_space.body_space.get(a=a)
            if global_nets is not None:
                agent_global_nets = global_nets[a]
            else:
                agent_global_nets = None
            agent = Agent(self.spec, body=body_a, a=a, agent_space=self, global_nets=agent_global_nets)
            self.agents.append(agent)
        logger.info(util.self_desc(self))

    def get(self, a):
        return self.agents[a]

    @lab_api
    def act(self, state_space):
        data_names = ('action',)
        action_v, = self.aeb_space.init_data_v(data_names)
        for agent in self.agents:
            a = agent.a
            state_a = state_space.get(a=a)
            action_a = agent.space_act(state_a)
            action_v[a, 0:len(action_a)] = action_a
        action_space, = self.aeb_space.add(data_names, (action_v,))
        return action_space

    @lab_api
    def update(self, state_space, action_space, reward_space, next_state_space, done_space):
        data_names = ('loss', 'explore_var')
        loss_v, explore_var_v = self.aeb_space.init_data_v(data_names)
        for agent in self.agents:
            a = agent.a
            state_a = state_space.get(a=a)
            action_a = action_space.get(a=a)
            reward_a = reward_space.get(a=a)
            next_state_a = next_state_space.get(a=a)
            done_a = done_space.get(a=a)
            loss_a, explore_var_a = agent.space_update(state_a, action_a, reward_a, next_state_a, done_a)
            loss_v[a, 0:len(loss_a)] = loss_a
            explore_var_v[a, 0:len(explore_var_a)] = explore_var_a
        loss_space, explore_var_space = self.aeb_space.add(data_names, (loss_v, explore_var_v))
        return loss_space, explore_var_space

    @lab_api
    def close(self):
        logger.info('AgentSpace.close')
        for agent in self.agents:
            agent.close()
