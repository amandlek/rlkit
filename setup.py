from setuptools import setup, find_packages
import sys

setup(name='rlkit',
      packages=[package for package in find_packages()
                if package.startswith('rlkit')],
      install_requires=['gtimer',
                        'torch',
                        'torchvision',
                        'python-dateutil',
                        'joblib'],
      description="Collection of reinforcement learning algorithms",
      author="Ajay Mandlekar",
      url='https://github.com/amandlek/rlkit',
      author_email="amandlek@stanford.edu",
      version="0.1.0")
