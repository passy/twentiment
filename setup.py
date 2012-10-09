#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


import twentiment


setup(
    name='twentiment',
    version=twentiment.__version__,
    description='Twitter sentiment analysis tool',
    long_description=open('README.rst').read(),
    author='Pascal Hartig',
    author_email='phartig@rdrei.net',
    url='https://github.com/passy/twentiment',
    packages=['twentiment', 'twentiment.thirdparty'],
    package_data={'': ['LICENSE', 'README.rst']},
    include_package_data=True,
    scripts=["bin/twentiment_server", "bin/twentiment_client"],
    install_requires=[
        'pyzmq',
        'six==1.2.0'
    ],
    license=open('LICENSE').read(),
    classifiers=(
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Operating System :: Unix',
        'Topic :: Communications',
        'Topic :: Internet :: WWW/HTTP'
    )
)
