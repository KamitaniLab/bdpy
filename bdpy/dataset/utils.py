"""Dataset utilities."""

from typing import Union

import hashlib
import urllib.request

from tqdm import tqdm


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
