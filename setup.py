import os
from setuptools import setup

setup(name='PyNMF',
      version='0.1',
      description='Python Non Negative Matrix Factorization Module',
      author='Diyah Utami',
      author_email='diyahutami97@gmail.com',
      url='https://github.com/diyahutami/pynmf',
      packages = ['pynmf'],    
      install_requires=['cvxopt', 'numpy', 'scipy'],
      long_description=open('README.md').read(),
      )     
