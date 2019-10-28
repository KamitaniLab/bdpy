'''
Setup script for bdpy

This file is a part of BdPy.
'''


from setuptools import setup


VERSION = 0.13


if __name__ == '__main__':

    # Setup
    setup(name='bdpy',
          version=VERSION,
          description='Brain decoder toolbox for Python',
          author='Shuntaro C. Aoki',
          author_email='brainliner-admin@atr.jp',
          maintainer='Shuntaro C. Aoki',
          maintainer_email='brainliner-admin@atr.jp',
          url='https://github.com/KamitaniLab/bdpy',
          license='MIT',
          packages=['bdpy',
                    'bdpy.bdata',
                    'bdpy.dataform',
                    'bdpy.distcomp',
                    'bdpy.fig',
                    'bdpy.ml',
                    'bdpy.mri',
                    'bdpy.preproc',
                    'bdpy.stats',
                    'bdpy.util'],
          install_requires=['numpy', 'scipy', 'scikit-learn', 'h5py'])
