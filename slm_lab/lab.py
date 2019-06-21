import os
import fire
import torch

from slm_lab.lib import logger, util
from slm_lab.experiment import search
from slm_lab.spec import spec_util
from slm_lab.experiment import search
from slm_lab.experiment.control import Session, Trial, Experiment

logger = logger.get_logger(__name__)

class Lab(object):
    "Mission Control for SLM Lab"
    
    def __init__(self):
#         "Does __init__ stuff"
        torch.set_num_threads(1)  # prevent multithread slowdown
        torch.multiprocessing.set_start_method('spawn', force=True)  # for distributed pytorch to work

    def enjoy(self, specfile, checkpoint):
        "Core operation of the Lab.  Runs an Agent in an Environment with optional on-screen rendering or video recording."
        logger.info(f'Running lab mode:enjoy with specfile:{specfile} checkpoint:{checkpoint}')
        spec = spec_util.get_eval_spec(specfile, checkpoint)
        # FIXME Why does this need to be in env?
        os.environ['lab_mode'] = 'enjoy'
        # spec = spec_util.override_enjoy_spec(spec)
        spec['meta']['max_session'] = 1
        Session(spec).run()

    def eval(self, specfile, checkpoint):
        "enjoy + computes metrics"
        logger.info(f'Running lab mode:eval with specfile:{specfile} checkpoint:{checkpoint}')
        spec = spec_util.get_eval_spec(specfile, checkpoint)
        # FIXME Why does this need to be in env?
        os.environ['lab_mode'] = 'eval'
        # spec = spec_util.override_eval_spec(spec)
        spec['meta']['max_session'] = 1
        # evaluate by episode is set in env clock init in env/base.py
        Session(spec).run()

    def train(self, specfile, specname):
        "eval + optimizes agent"
        logger.info(f'Running lab mode:train with specfile:{specfile} specname:{specname}')
        spec = spec_util.get(specfile, specname)
        # FIXME Why does this need to be in env?
        os.environ['lab_mode'] = 'train'
        spec_util.save(spec)  # first save the new spec
        spec_util.tick(spec, 'trial')
        Trial(spec).run()

    def dev(self, specfile, specname):
        "train + limit the number of trials & sessions. Useful for iterative development."
        logger.info(f'Running lab mode:dev with specfile:{specfile} specname:{specname}')
        spec = spec_util.get(specfile, specname)
        # FIXME Why does this need to be in env?
        os.environ['lab_mode'] = 'dev'
        spec_util.save(spec)  # first save the new spec
        # spec = spec_util.override_dev_spec(spec)
        spec['meta']['max_session'] = 1
        spec['meta']['max_trial'] = 2
        spec_util.tick(spec, 'trial')
        Trial(spec).run()
        
    def search(self, specfile, specname):
        "runs train mode multiple times across the parameterized specs"
        logger.info(f'Running lab mode:search with specfile:{specfile} specname:{specname}')
        spec = spec_util.get(specfile, specname)
        # assert 'spec_params' in spec
        param_specs = spec_util.get_param_specs(spec)
        search.run_param_specs(param_specs)
 
    # FIXME I don't think we really need this unless it does something more advanced than
    #       what you could accomplish with a simple bash script that runs the lab a few times.
    def jobs(self, spec='job/experiments.json'):
        "runs a set of distributed jobs in the lab (any list of lab commands)"
        print(f'Running jobs spec: {spec}...')
        for cmd in util.read(spec):
            print(f'Running job: {cmd}')
            fire.Fire(Lab, cmd)
#             for spec_name, lab_mode in spec_and_mode.items():
#                 fire.Fire(Lab, )
#                 read_spec_and_run(spec_file, spec_name, lab_mode)


        
        