"""Dataset utilities."""

from typing import List, TypedDict, Union

import hashlib
import inspect
import os
import subprocess
import urllib.request

from tqdm import tqdm


class FileDict(TypedDict):
    name: str
    url: str
    md5sum: str


def download_file(url: str, destination: str, progress_bar: bool = True, md5sum: Union[str, None] = None) -> None:
    """Download a file.

    Parameters
    ----------
    url: str
      File URL.
    destination: str
      Path to save the file.
    progress_bar: bool = True
      Show progress bar if True.
    md5sum: Union[str, None] = None
      md5sum hash of the file.

    Returns
    -------
    None
    """
    response = urllib.request.urlopen(url)
    file_size = int(response.info()["Content-Length"])

    def __show_progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            progress_bar.update(downloaded - progress_bar.n)

    with tqdm(total=file_size, unit='B', unit_scale=True, desc=destination, ncols=100) as progress_bar:
        urllib.request.urlretrieve(url, destination, __show_progress)

    if md5sum is not None:
        md5_hash = hashlib.md5()
        with open(destination, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        md5sum_test = md5_hash.hexdigest()
        if md5sum != md5sum_test:
            raise ValueError(f'md5sum mismatch. \nExpected: {md5sum}\nActual: {md5sum_test}')


def download_splitted_file(file_list: List[FileDict], destination: str, progress_bar: bool = True, md5sum: Union[str, None] = None) -> None:
    """Download a file.

    Parameters
    ----------
    file_list: List[FileDict]
      List of split files.
    destination: str
      Path to save the file.
    progress_bar: bool = True
      Show progress bar if True.
    md5sum: Union[str, None] = None
      md5sum hash of the file.

    Returns
    -------
    None
    """
    wdir = os.path.dirname(destination)

    # Download split files
    for sf in file_list:
        _output = os.path.join(wdir, sf['name'])
        if not os.path.exists(_output):
            print(f'Downloading {_output} from {sf["url"]}')
            download_file(sf['url'], _output, progress_bar=progress_bar, md5sum=sf['md5sum'])

    # Merge files
    subprocess.run(f"cat {destination}-* > {destination}", shell=True)
    print(f"File created: {destination}")

    # Check md5sum
    if md5sum is not None:
        md5_hash = hashlib.md5()
        with open(destination, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        md5sum_test = md5_hash.hexdigest()
        if md5sum != md5sum_test:
            raise ValueError(f'md5sum mismatch. \nExpected: {md5sum}\nActual: {md5sum_test}')
