'''
Setup script for bdpy

This file is a part of BdPy.
'''


from setuptools import setup


VERSION_FILE = 'version'


if __name__ == '__main__':

    # Get version num
    with open(VERSION_FILE, 'r') as f:
        version_num = f.readline().strip()

    # Setup
    setup(name='bdpy',
          version=version_num,
          description='Brain decoder toolbox',
          author='Shuntaro Aoki',
          author_email='brainliner-admin@atr.jp',
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
