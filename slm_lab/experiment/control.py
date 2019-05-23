'''
The control module
Creates and controls the units of SLM lab: Experiment, Trial, Session
'''
from copy import deepcopy
from importlib import reload
from slm_lab.agent import AgentSpace, Agent
from slm_lab.agent.net import net_util
from slm_lab.env import EnvSpace, make_env
from slm_lab.experiment import analysis, retro_analysis, search
from slm_lab.experiment.monitor import AEBSpace, Body, enable_aeb_space
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import os
import torch.multiprocessing as mp


class Session:
    '''
    The base lab unit to run a RL session for a spec.
    Given a spec, it creates the agent and env, runs the RL loop,
    then gather data and analyze it to produce session data.
    '''

    def __init__(self, spec, global_nets=None):
        self.spec = spec
        self.index = self.spec['meta']['session']
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        util.set_logger(self.spec, logger, 'session')
        analysis.save_spec(spec, unit='session')
        self.data = None

        # init agent and env
        self.env = make_env(self.spec)
        with util.ctx_lab_mode('eval'):  # env for eval
            self.eval_env = make_env(self.spec)
        body = Body(self.env, self.spec['agent'])
        self.agent = Agent(self.spec, body=body, global_nets=global_nets)

        enable_aeb_space(self)  # to use lab's data analysis framework
        logger.info(util.self_desc(self))

    def to_ckpt(self, env, mode='eval'):
        '''Check with clock whether to run log/eval ckpt: at the start, save_freq, and the end'''
        clock = env.clock
        tick = clock.get()
        if mode == 'eval' and util.in_eval_lab_modes():  # avoid double-eval: eval-ckpt in eval mode
            return False
        frequency = env.eval_frequency if mode == 'eval' else env.log_frequency
        if mode == 'log' and tick == 0:  # avoid log ckpt at init
            to_ckpt = False
        elif frequency is None:  # default episodic
            to_ckpt = env.done
        elif clock.max_tick_unit == 'epi' and not env.done:  # epi ckpt needs env done
            to_ckpt = False
        else:  # normal ckpt condition by mod remainder (general for venv)
            rem = env.num_envs or 1
            to_ckpt = (tick % frequency < rem) or tick == clock.max_tick
        return to_ckpt

    def try_ckpt(self, agent, env):
        '''Check then run checkpoint log/eval'''
        if self.to_ckpt(env, 'log'):
            agent.body.train_ckpt()
            agent.body.log_summary('train')

        if self.to_ckpt(env, 'eval'):
            total_reward = self.run_eval()
            agent.body.eval_ckpt(self.eval_env, total_reward)
            agent.body.log_summary('eval')
            if analysis.new_best(agent):
                agent.save(ckpt='best')
            if env.clock.get() > 0:  # nothing to analyze at start
                analysis.analyze_session(self, eager_analyze_trial=True)

    def run_eval(self):
        with util.ctx_lab_mode('eval'):  # enter eval context
            self.agent.algorithm.update()  # set explore_var etc. to end_val under ctx
            self.eval_env.clock.tick('epi')
            state = self.eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                self.eval_env.clock.tick('t')
                action = self.agent.act(state)
                next_state, reward, done, info = self.eval_env.step(action)
                state = next_state
                total_reward += reward
        # exit eval context, restore variables simply by updating
        self.agent.algorithm.update()
        return total_reward

    def run_rl(self):
        '''Run the main RL loop until clock.max_tick'''
        logger.info(f'Running RL loop for trial {self.spec["meta"]["trial"]} session {self.index}')
        clock = self.env.clock
        state = self.env.reset()
        done = False
        while True:
            if util.epi_done(done):  # before starting another episode
                self.try_ckpt(self.agent, self.env)
                if clock.get() < clock.max_tick:  # reset and continue
                    clock.tick('epi')
                    state = self.env.reset()
                    done = False
            self.try_ckpt(self.agent, self.env)
            if clock.get() >= clock.max_tick:  # finish
                break
            clock.tick('t')
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state

    def close(self):
        '''Close session and clean up. Save agent, close env.'''
        self.agent.close()
        self.env.close()
        self.eval_env.close()
        logger.info('Session done and closed.')

    def run(self):
        self.run_rl()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


class SpaceSession(Session):
    '''Session for multi-agent/env setting'''

    def __init__(self, spec, global_nets=None):
        self.spec = spec
        self.index = self.spec['meta']['session']
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        util.set_logger(self.spec, logger, 'session')
        analysis.save_spec(spec, unit='session')
        self.data = None

        self.aeb_space = AEBSpace(self.spec)
        self.env_space = EnvSpace(self.spec, self.aeb_space)
        self.aeb_space.init_body_space()
        self.agent_space = AgentSpace(self.spec, self.aeb_space, global_nets)

        logger.info(util.self_desc(self))

    def try_ckpt(self, agent_space, env_space):
        '''Try to checkpoint agent at the start, save_freq, and the end'''
        # TODO ckpt and eval not implemented for SpaceSession
        pass
        # for agent in agent_space.agents:
        #     for body in agent.nanflat_body_a:
        #         env = body.env
        #         super().try_ckpt(agent, env)

    def run_all_episodes(self):
        '''
        Continually run all episodes, where each env can step and reset at its own clock_speed and timeline.
        Will terminate when all envs done are done.
        '''
        all_done = self.aeb_space.tick('epi')
        state_space = self.env_space.reset()
        while not all_done:
            self.try_ckpt(self.agent_space, self.env_space)
            all_done = self.aeb_space.tick()
            action_space = self.agent_space.act(state_space)
            next_state_space, reward_space, done_space, info_v = self.env_space.step(action_space)
            self.agent_space.update(state_space, action_space, reward_space, next_state_space, done_space)
            state_space = next_state_space
        self.try_ckpt(self.agent_space, self.env_space)

    def close(self):
        '''Close session and clean up. Save agent, close env.'''
        self.agent_space.close()
        self.env_space.close()
        logger.info('Session done and closed.')

    def run(self):
        self.run_all_episodes()
        self.data = analysis.analyze_session(self, tmp_space_session_sub=True)  # session fitness
        self.close()
        return self.data


def init_run_session(*args):
    '''Runner for multiprocessing'''
    session = Session(*args)
    return session.run()


def init_run_space_session(*args):
    '''Runner for multiprocessing'''
    session = SpaceSession(*args)
    return session.run()


class Trial:
    '''
    The lab unit which runs repeated sessions for a same spec, i.e. a trial
    Given a spec and number s, trial creates and runs s sessions,
    then gathers session data and analyze it to produce trial data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.index = self.spec['meta']['trial']
        util.set_logger(self.spec, logger, 'trial')
        analysis.save_spec(spec, unit='trial')
        self.session_data_dict = {}
        self.data = None

        self.is_singleton = spec_util.is_singleton(spec)  # singleton mode as opposed to multi-agent-env space
        self.SessionClass = Session if self.is_singleton else SpaceSession
        self.mp_runner = init_run_session if self.is_singleton else init_run_space_session

    def parallelize_sessions(self, global_nets=None):
        workers = []
        for _s in range(self.spec['meta']['max_session']):
            spec_util.tick(self.spec, 'session')
            w = mp.Process(target=self.mp_runner, args=(deepcopy(self.spec), global_nets))
            w.start()
            workers.append(w)
        for w in workers:
            w.join()
        session_datas = retro_analysis.session_data_dict_for_dist(self.spec)
        return session_datas

    def run_sessions(self):
        logger.info('Running sessions')
        if util.get_lab_mode() in ('train', 'eval') and self.spec['meta']['max_session'] > 1:
            # when training a single spec over multiple sessions
            session_datas = self.parallelize_sessions()
        else:
            session_datas = []
            for _s in range(self.spec['meta']['max_session']):
                spec_util.tick(self.spec, 'session')
                session = self.SessionClass(deepcopy(self.spec))
                session_data = session.run()
                session_datas.append(session_data)
                if analysis.is_unfit(session_data, session):
                    break
        return session_datas

    def init_global_nets(self):
        session = self.SessionClass(deepcopy(self.spec))
        if self.is_singleton:
            session.env.close()  # safety
            global_nets = net_util.init_global_nets(session.agent.algorithm)
        else:
            session.env_space.close()  # safety
            global_nets = [net_util.init_global_nets(agent.algorithm) for agent in session.agent_space.agents]
        return global_nets

    def run_distributed_sessions(self):
        logger.info('Running distributed sessions')
        global_nets = self.init_global_nets()
        session_datas = self.parallelize_sessions(global_nets)
        return session_datas

    def close(self):
        logger.info('Trial done and closed.')

    def run(self):
        if self.spec['meta'].get('distributed') == False:
            session_datas = self.run_sessions()
        else:
            session_datas = self.run_distributed_sessions()
        self.session_data_dict = {data.index[0]: data for data in session_datas}
        self.data = analysis.analyze_trial(self)
        self.close()
        return self.data


class Experiment:
    '''
    The lab unit to run experiments.
    It generates a list of specs to search over, then run each as a trial with s repeated session,
    then gathers trial data and analyze it to produce experiment data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.index = self.spec['meta']['experiment']
        util.set_logger(self.spec, logger, 'trial')
        analysis.save_spec(spec, unit='experiment')
        self.trial_data_dict = {}
        self.data = None
        SearchClass = getattr(search, spec['meta'].get('search'))
        self.search = SearchClass(self)

    def init_trial_and_run(self, spec):
        '''Method to run trial with the properly updated spec (trial_index) from experiment.search.lab_trial.'''
        trial = Trial(spec)
        trial_data = trial.run()
        return trial_data

    def close(self):
        reload(search)  # fixes ray consecutive run crashing due to bad cleanup
        logger.info('Experiment done and closed.')

    def run(self):
        self.trial_data_dict = self.search.run()
        self.data = analysis.analyze_experiment(self)
        self.close()
        return self.data
