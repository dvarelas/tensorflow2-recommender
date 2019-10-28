from setuptools import setup

with open('requirements.txt', 'r') as fp:
    requirements = [x.strip() for x in fp.readlines()]

setup(name='tensorflow2-recommender',
      version='0.1.0',
      description='Recommender systems using TF2.0',
      author='Dionysis Varelas',
      install_requires=requirements)
