# BdPy_mtanaka

[bdpy](https://github.com/KamitaniLab/bdpy) より下記の点を変更した．

- `PyFastL2iR_mtanaka` に対応し，unit ごとに選択された voxel の index を保存する．

## License

[![GitHub license](https://img.shields.io/github/license/KamitaniLab/bdpy)](https://github.com/KamitaniLab/bdpy/blob/master/LICENSE)

Python package for brain decoding analysis

## Requirements

- Python 3.5, or later
- numpy
- scipy
- scikit-learn
- h5py
- hdf5storage
- pyyaml

### Optional requirements

- `dataform` module
    - pandas
- `dl.caffe` module
    - Caffe
    - Pillow
    - tqdm
- `dl.torch` module
    - PyTorch
    - Pillow
- `fig` module
    - matplotlib
    - Pillow
- `mri` module
    - nipy
    - nibabel
    - pandas
- `recon.torch` module
    - PyTorch
    - Pillow

