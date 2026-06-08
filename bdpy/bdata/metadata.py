"""MetaData class.

This file is a part of BdPy
"""


from typing import Callable, List, Optional, Sequence, Union, overload

import numpy as np
from typing_extensions import Literal

MetaDataSetValue = Optional[Union[np.ndarray, Sequence[float]]]
MetaDataUpdater = Callable[[np.ndarray, np.ndarray], Union[np.ndarray, Sequence[float]]]


class MetaData(object):
    """MetaData class.

    'MetaData' is a list of dictionaries. Each element has three keys: 'key',
    'value', and 'description'.
    """

    def __init__(
        self,
        key: Optional[List[str]] = None,
        value: Optional[np.ndarray] = None,
        description: Optional[List[str]] = None,
    ) -> None:
        if key is None:
            key = []
        if value is None:
            value = np.ndarray((0, 0), dtype=float)
        if description is None:
            description = []

        self.__key = key
        self.__value = value
        self.__description = description

    @property
    def key(self) -> List[str]:
        """Meta-data keys."""
        return self.__key

    @key.setter
    def key(self, x: List[str]) -> None:
        self.__key = x

    @property
    def value(self) -> np.ndarray:
        """Meta-data values."""
        return self.__value

    @value.setter
    def value(self, x: np.ndarray) -> None:
        self.__value = x

    @property
    def description(self) -> List[str]:
        """Meta-data descriptions."""
        return self.__description

    @description.setter
    def description(self, x: List[str]) -> None:
        self.__description = x

    def set(
        self,
        key: str,
        value: MetaDataSetValue,
        description: str,
        updater: Optional[MetaDataUpdater] = None,
    ) -> None:
        """Set meta-data with `key`, `description`, and `value`.

        Parameters
        ----------
        key : str
            Meta-data key
        value : array_like
            Meta-data value
        description : str
            Meta-data description
        updater : function
            Function applied to meta-data value when meta-data named `key` already exists.
            It should take two args: new and old meta-data values.
        """
        # If `value` is None, `set` does not update the value.
        is_novalue = True if value is None else False

        value_array = np.array(value)

        if key in self.__key:
            # Update existing metadata

            indices = [i for i, k in enumerate(self.__key) if k == key]

            if len(indices) > 1:
                raise ValueError('Multiple meta-data with the same key is not supported')

            ind = indices[0]

            self.__description[ind] = description

            # If `value` is None, `set` does not update the value.
            if is_novalue:
                return None

            if value_array.shape[0] > self.get_value_len():
                cols = np.empty((self.__value.shape[0], value_array.shape[0] - self.get_value_len()))
                cols[:] = np.nan

                self.__value = np.hstack([self.__value, cols])

            if updater is None:
                self.__value[ind, :] = value_array
            else:
                self.__value[ind, :] = np.array(updater(value_array, self.__value[ind, :]), dtype=float)
        else:
            # Add new metadata
            self.__key.append(key)
            self.__description.append(description)

            if value_array.shape[0] > self.get_value_len():
                cols = np.empty((self.__value.shape[0], value_array.shape[0] - self.get_value_len()))
                cols[:] = np.nan

                self.__value = np.hstack([self.__value, cols])

            self.__value = np.vstack([self.__value, value_array])


    @overload
    def get(self, key: str, field: Literal['value']) -> Optional[np.ndarray]:
        ...

    @overload
    def get(self, key: str, field: Literal['description']) -> Optional[str]:
        ...

    @overload
    def get(self, key: str, field: str) -> Union[np.ndarray, str, None]:
        ...

    def get(self, key: str, field: str) -> Union[np.ndarray, str, None]:
        """Return meta-data specified by `key`.

        Parameters
        ----------
        key : str
            Meta-data key
        field : str
            Field name of meta-data (either 'value' or 'description')

        Returns
        -------
        array, str or None
            Meta-data value or description. If `key` was not found in
            the metadata, `None` is returned.
        """
        if key in self.__key:
            ind = self.__key.index(key)
        else:
            return None

        if field == 'value':
            return self.__value[ind, :].astype(float)

        if field == 'description':
            return self.__description[ind]

        return None


    def get_value_len(self) -> int:
        """Return length of meta-data value."""
        return self.__value.shape[1]


    def keylist(self) -> List[str]:
        """Return a list of keys."""
        return self.__key
