import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

test_args = [
    '--verbose',
    '--capture=sys',
    '--log-level=INFO',
    '--log-cli-level=INFO',
    '--log-file-level=INFO',
    '--no-flaky-report',
    '--timeout=300',
    '--cov-report=html',
    '--cov-report=term',
    '--cov-report=xml',
    '--cov=slm_lab',
    '--ignore=test/spec/test_dist_spec.py',
    'test',
]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        os.environ['PY_ENV'] = 'test'
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)
        
        
here = path.abspath(path.dirname(__file__))
def get_install_requires_from_pipfiles():
    import json
    with open(os.path.join(here, 'Pipfile.lock'), encoding='utf-8') as f:
        pipfile_lock = json.load(f)
    with open(os.path.join(here, 'Pipfile'), encoding='utf-8') as f:
        pipfile = f.read()

    def get_pkg_str(pkgname, pkg_lock):
        pkg_str = pkgname
        if 'extras' in pkg_lock:
            extras = ''.join([f'[{extra}]' for extra in pkg_lock['extras']])
            pkg_str += extras
        pkg_str += pkg_lock['version']
        return pkg_str

    install_requires = [
        get_pkg_str(pkgname, pkg)
    for pkgname, pkg in pipfile_lock['default'].items() if pkgname in pipfile]
    return install_requires


setup(
    name='slm_lab',
    version='4.0.0',
    description='Modular Deep Reinforcement Learning framework in PyTorch.',
    keywords='SLM Lab',
    url='https://github.com/kengz/slm_lab',
    author='kengz,lgraesser',
    author_email='kengzwl@gmail.com',
    license='MIT',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'slm_lab = slm_lab.main'
        ]
    },
    install_requires=get_install_requires_from_pipfiles(),
    zip_safe=False,
    include_package_data=True,
    dependency_links=[],
    extras_require={
        'dev': [],
        'docs': [],
        'testing': []
    },
    classifiers=[],
    test_suite='test',
    cmdclass={'test': PyTest},
)
