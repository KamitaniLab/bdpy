# coding: utf-8
"""
MetaData class

This file is a part of BdPy
"""


import numpy as np


class MetaData(list):
    """
    MetaData class

    'MetaData' is a list of dictionaries. Each element is a dictionary which
    has three keys: 'key', 'value', and 'description'.
    """

    def __init__(self):
        list.__init__(self)

    def set(self, key, value, description, updater=None):
        """
        Set meta-data with `key`, `description`, and `value`

        Parameters
        ----------
        key : str
            Meta-data key
        value
            Meta-data value
        description : str
            Meta-data description
        updater : function
            Function applied to meta-data value when meta-data named `key` already exists.
            It takes two args: new and old meta-data values.
        """

        if value is not None and (len(value) > self.get_value_len()):
            self.extend_value(len(value) - self.get_value_len())

        key_list = self.keylist()
        key_hit_count = key_list.count(key)

        if key_hit_count == 0:
            # Add new meta-data
            self.append({'key' : key,
                         'description' : description,
                         'value' : np.array(value, dtype=np.float)})
        elif key_hit_count == 1:
            # Update existing meta-data with 'value'
            ind = key_list.index(key)

            self[ind]['description'] = description

            if updater is None:
                self[ind]['value'] = np.array(value, dtype=np.float)
            else:
                self[ind]['value'] = np.array(updater(value, self[ind]['value']), dtype=np.float)
        else:
            raise ValueError('Multiple meta-data with the same key is not supported')


    def extend_value(self, num_add):
        """
        Add columns to meta-data value matrix

        Parameters
        ----------
        num_add
            Num of columns added to meta-data
        """

        for i in xrange(len(self)):
            self[i]['value'] = np.hstack((self[i]['value'],
                                          [np.nan for _ in xrange(num_add)]))


    def get(self, key, field):
        """
        Returns meta-data specified by `key`


        Parameters
        ----------
        key : str
            Meta-data key
        field : str
            Field name of meta-data (either 'value' or 'description')

        Returns
        ----------
            Meta-data value or description (str)
        """

        key_list = self.keylist()
        key_hit_count = key_list.count(key)

        if key_hit_count == 0:
            return None
        elif key_hit_count == 1:
            ind = key_list.index(key)
            return self[ind][field]
        else:
            raise ValueError('Multiple meta-data with the same key is not supported')


    def get_value_len(self):
        """Returns length of meta-data value"""

        if self:
            return len(self[0]['value'])
        else:
            return 0


    def keylist(self):
        """Returns a list of keys"""

        return [m['key'] for m in self]
