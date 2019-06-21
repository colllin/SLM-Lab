# The SLM Lab entrypoint
# to run scheduled set of specs:
# python run_lab.py jobs job/experiments.json
# to run a single spec:
# python run_lab.py train slm_lab/spec/experimental/a2c_pong.json a2c_pong
import sys
import fire
from xvfbwrapper import Xvfb
from slm_lab.lib import logger
from slm_lab.lab import Lab

debug_modules = [
    # 'algorithm',
]
debug_level = 'DEBUG'
logger.toggle_debug(debug_modules, debug_level)
logger = logger.get_logger(__name__)


def main():
    '''Main method to run jobs from scheduler or from a spec directly'''
    if sys.platform == 'darwin':
        # avoid xvfb on MacOS: https://github.com/nipy/nipype/issues/1400
        fire.Fire(Lab)
    else:
        with Xvfb() as xvfb:  # safety context for headless machines
            fire.Fire(Lab)


if __name__ == '__main__':
    main()
