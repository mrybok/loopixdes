from setuptools import setup

setup(
    name='LoopixDES',
    url='https://github.com/mrybok/loopixdes',
    author='Marcin Rybok',
    author_email='s1766172@ed.ac.uk',
    packages=['loopixdes'],
    install_requires=['gym', 'tqdm', 'numpy', 'simpy', 'tensorflow'],
    version='0.1',
    license='MIT',
    description='Minimal implementation of the Loopix mix network design for simulating traffic '
                'in the network and optimizing its parameters.',
    long_description=open('README.md').read()
)
