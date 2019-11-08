'''
Setup script for bdpy

This file is a part of BdPy.
'''


from setuptools import setup


VERSION = '0.14rc3'

if __name__ == '__main__':

    # Long description
    with open('./README.md') as f:
        long_description = f.read()

    # Setup
    setup(name='bdpy',
          version=VERSION,
          description='Brain decoder toolbox for Python',
          long_description=long_description,
          long_description_content_type='text/markdown',
          author='Shuntaro C. Aoki',
          author_email='brainliner-admin@atr.jp',
          maintainer='Shuntaro C. Aoki',
          maintainer_email='brainliner-admin@atr.jp',
          url='https://github.com/KamitaniLab/bdpy',
          license='MIT',
          keywords='neuroscience, neuroimaging, brain decoding, fmri, machine learning',
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
