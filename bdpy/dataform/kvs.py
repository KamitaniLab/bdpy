"""Key-value store."""


from typing import List, Tuple, Union, Optional

import os
import sqlite3
from pathlib import Path

import numpy as np


_array_t = np.ndarray
_path_t = Union[str, Path]


class BaseKeyValueStore(object):
    """Base class for key-value store."""

    def get(self, **kwargs) -> _array_t:
        raise NotImplementedError("get should be implemented in the subclass.")

    def set(self, value: _array_t, **kwargs) -> None:
        raise NotImplementedError("set should be implemented in the subclass.")


class SQLite3KeyValueStore(BaseKeyValueStore):
    """Key-value store implemented on SQLite3."""

    def __init__(self, path: _path_t, timeout: int = 60, keys: Optional[List[str]] = None):
        """Initialize the SQLite3KeyValueStore object.

        Parameters
        ---------- 
        path:  _path_t
          The path to the SQLite database file.
        keys:  List[str], optional
          The list of keys. Defaults to [].

        """
        if keys is None:
            keys = []
        self._path = path
        self._keys = keys

        new_db = not os.path.exists(self._path)

        # Connect to DB
        self._conn = sqlite3.connect(self._path, isolation_level='EXCLUSIVE', timeout=timeout)

        # Enable foreign key
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA foreign_keys = true")
        cursor.close()

        # Initialize DB
        if new_db:
            self._init_empty_db()
            if not keys:
                raise ValueError("Keys must be given when creating a new database.")
            # Add keys
            for key in keys:
                self._add_key(key)
        else:
            self._validate_db(keys)
            self._keys = self._get_keys()

    def set(self, value: _array_t, **kwargs) -> None:
        """
        Set value for the given keys.
        Transaction mode: DEFERRED
        """
        # Check if the keys are valid
        for key in kwargs.keys():
            if key not in self._keys:
                raise ValueError(f"Key '{key}' is not defined.")

        # Check if all keys are given
        if len(kwargs) != len(self._keys):
            raise ValueError("All keys must be given.")

        # Check if the given keys already exist
        key_group_id = self._get_key_group_id(**kwargs)
        cursor = self._conn.cursor()
        cursor.execute("BEGIN DEFERRED;")
        if key_group_id is None:
            # Add new key-value pair
            sql = "INSERT INTO key_value_store (value) VALUES (?)"            
            cursor.execute(sql, (value.astype(float).tobytes(order='C'),))
            key_value_store_id = cursor.lastrowid
            self._add_key_group_id(key_value_store_id, **kwargs)
        else:
            # Update existing key-value pair
            sql = f"""
            UPDATE key_value_store
            SET value = ?
            WHERE id = {key_group_id}
            """
            cursor.execute(sql, (value.astype(float).tobytes(order='C'),))
        self._conn.commit()
        cursor.close()

        return None

    def lock(self, **kwargs) -> bool:
        """
        If a record with the specified condition does not exist, insert a record with a null value and return True. 
        If a record exists, return False.
        Transaction mode: EXCLUSIVE
        """
        # Check if the keys are valid
        for key in kwargs.keys():
            if key not in self._keys:
                raise ValueError(f"Key '{key}' is not defined.")

        # Check if all keys are given
        if len(kwargs) != len(self._keys):
            raise ValueError("All keys must be given.")

        # Start EXCLUSIVE transaction
        cursor = self._conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE;")

        # Check if a record with the specified condition already exists
        try:
            key_value_store_id = self._get_key_group_id(**kwargs)
        except ValueError:
            # Close transaction
            self._conn.commit()
            cursor.close()
            raise

        # If the condition already exists,
        # It is determined that it is impossible to obtain the lock,
        # close the cursor, return False.
        if key_value_store_id is not None:
            # Close transaction
            self._conn.commit()
            cursor.close()
            return False

        # If no record with the specified condition exists,
        # take a lock.
        # Add new record to key-value pair and get the last key_value_store_id
        sql = "INSERT INTO key_value_store (value) VALUES (?)"
        cursor.execute(sql, (np.array([[]]).astype(float).tobytes(order='C'),))
        key_value_store_id = cursor.lastrowid
        self._add_key_group_id(key_value_store_id, **kwargs)

        # Close transaction
        self._conn.commit()
        cursor.close()

        # Lock をとることに成功したので True を返す
        return True

    def get(self, **kwargs) -> Optional[_array_t]:
        """Get value for the given keys."""
        key_group_id = self._get_key_group_id(**kwargs)
        if key_group_id is None:
            return None
        sql = f"""
        SELECT key_value_store.value FROM key_value_store
        WHERE key_value_store.id = {key_group_id}
        """
        cursor = self._conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        if len(res) == 0:
            return None
        elif len(res) > 1:
            raise ValueError("Multiple values found.")
        else:
            return np.frombuffer(res[0][0], dtype=float)

    def exists(self, **kwargs) -> bool:
        """Check if the key-value pair exists."""
        return self._get_key_group_id(**kwargs) is not None

    def delete(self, **kwargs) -> None:
        """Delete the key-value pair."""
        key_group_id = self._get_key_group_id(**kwargs)
        if key_group_id is None:
            return None

        # Delete from key_group_members and key_value_store
        sqls = [
            f"""
            DELETE FROM key_group_members WHERE key_value_store_id = {key_group_id}
            """,
            f"""
            DELETE FROM key_value_store WHERE id = {key_group_id}
            """,
        ]
        # Start transaction        
        cursor = self._conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE;")
        for sql in sqls:
            cursor.execute(sql)
        # Close transaction
        self._conn.commit()
        cursor.close()
        return None

    def _add_key_group_id(self, key_value_store_id: int, **kwargs) -> int:
        """Add key group ID."""
        # Open cursor
        cursor = self._conn.cursor()

        for key, inst in kwargs.items():
            # Add key instance if not exists
            key_instance_id = self._get_key_instance_id(key, inst)
            if key_instance_id is not None:
                continue
            key_name_id = self._get_key_name_id(key)
            sql = f"""
            INSERT OR IGNORE INTO key_instances (name, key_name_id) VALUES ('{inst}', {key_name_id})
            """
            cursor.execute(sql)

        inst_ids = [self._get_key_instance_id(key, inst) for key, inst in kwargs.items()]
        sqls = [
            f"INSERT INTO key_group_members (key_value_store_id, key_instance_id) VALUES ({key_value_store_id}, {inst_id})"
            for inst_id in inst_ids
        ]
        for sql in sqls:
            cursor.execute(sql)

        # Close cursor
        cursor.close()
        return key_value_store_id

    def _get_key_group_id(self, **kwargs) -> Optional[int]:
        """Get key group ID."""
        where = self._generate_where(**kwargs)
        sql = f"""
        SELECT kgm.key_value_store_id
        FROM key_group_members AS kgm
        JOIN key_instances ON kgm.key_instance_id = key_instances.id
        JOIN key_names     ON key_instances.key_name_id = key_names.id
        WHERE
        {where}
        GROUP BY kgm.key_value_store_id
        """
        cursor = self._conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        if len(res) == 0:
            # Not found
            return None
        elif len(res) > 1:
            raise ValueError("Multiple key groups found.")
        else:
            return res[0][0]

    def _add_key(self, key: str) -> None:
        sql = f"INSERT OR IGNORE INTO key_names (name) VALUES('{key}')"
        cursor = self._conn.cursor()
        cursor.execute(sql)
        self._conn.commit()
        cursor.close()
        return None

    def _get_keys(self) -> List[str]:
        sql = "SELECT name FROM key_names"
        cursor = self._conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        return [r[0] for r in res]

    def _get_key_name_id(self, key: str) -> int:
        """Get key name ID."""
        sql = f"SELECT id FROM key_names WHERE name = '{key}'"
        cursor = self._conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        if not res:
            raise ValueError(f"Key '{key}' is not defined.")
        return res[0][0]

    def _get_key_instance_id(self, key: str, inst: str) -> Optional[int]:
        """Get key instance ID."""
        key_name_id = self._get_key_name_id(key)
        sql = f"SELECT id FROM key_instances WHERE name = '{inst}' AND key_name_id = {key_name_id}"
        cursor = self._conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        if not res:
            return None
        return res[0][0]

    def _init_empty_db(self) -> None:
        """Create empty tables."""
        sqls = [
            """
            CREATE TABLE IF NOT EXISTS key_names (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT UNIQUE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS key_instances (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT UNIQUE,
              key_name_id INTEGER,
              FOREIGN KEY (key_name_id) REFERENCES key_names(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS key_value_store (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              value BLOB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS key_group_members (
              key_value_store_id INTEGER,
              key_instance_id INTEGER,
              PRIMARY KEY (key_value_store_id, key_instance_id),
              FOREIGN KEY (key_value_store_id) REFERENCES key_value_store(id),
              FOREIGN KEY (key_instance_id) REFERENCES key_instances(id)
            )
            """,
        ]
        cursor = self._conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE;")
        for sql in sqls:
            cursor.execute(sql)
        self._conn.commit()
        cursor.close()
        return None

    def _validate_db(self, keys: List[str]) -> None:
        pass

    def _generate_where(self, **kwargs) -> str:
        """Generate WHERE clause."""
        where = ' AND '.join(
            [
               f"""
               EXISTS(
                 SELECT * FROM key_group_members AS kgm{i}
                 JOIN key_instances AS ki{i} ON kgm{i}.key_instance_id = ki{i}.id
                 JOIN key_names     AS kn{i} ON ki{i}.key_name_id = kn{i}.id
                 WHERE
                   kgm.key_value_store_id = kgm{i}.key_value_store_id
                   AND
                   kn{i}.name = '{key}' AND ki{i}.name = '{inst}'
               )
               """
               for i, (key, inst) in enumerate(kwargs.items())
               ]
        )
        return where
