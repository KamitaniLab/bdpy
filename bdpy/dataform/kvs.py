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

    def __init__(self, path: _path_t, keys: Optional[List[str]] = None):
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
        self._conn = sqlite3.connect(self._path, isolation_level="EXCLUSIVE")

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
        """Set value for the given keys."""
        # Check if the keys are valid
        for key in kwargs.keys():
            if key not in self._keys:
                raise ValueError(f"Key '{key}' is not defined.")

        # Check if all keys are given
        if len(kwargs) != len(self._keys):
            raise ValueError("All keys must be given.")

        # Set transaction
        self._conn.execute("BEGIN TRANSACTION;")
        _v = value.astype(float).tobytes(order='C')
        where = self._generate_where(**kwargs)
        insert_instances = ', '.join([
            f"('{inst}', (SELECT id FROM key_names WHERE name = '{key}'))"
            for key, inst in kwargs.items()
        ])
        insert_members = ', '.join([
            f"((SELECT id FROM key_value_store WHERE rowid = (SELECT * FROM kvs_last_inserted_rowid)), (SELECT ki.id FROM key_instances AS ki JOIN key_names AS kn ON ki.key_name_id = kn.id WHERE kn.name = '{key}' AND ki.name = '{inst}'))"
            for key, inst in kwargs.items()
        ])
        sqls_prep = f"""
        CREATE TABLE tmp AS
        WITH hit AS (
            SELECT kgm.key_value_store_id FROM key_group_members AS kgm
            JOIN key_instances            AS ki ON kgm.key_instance_id = ki.id
            JOIN key_names                AS kn ON ki.key_name_id = kn.id
        WHERE
            {where}
        GROUP BY kgm.key_value_store_id
        )
        SELECT * FROM hit;

        CREATE TABLE kvs_last_inserted_rowid (rowid INTEGER);
        CREATE TRIGGER kvs_insert
        AFTER INSERT ON key_value_store
        BEGIN
            DELETE FROM kvs_last_inserted_rowid;
            INSERT INTO kvs_last_inserted_rowid (rowid) VALUES (new.rowid);
        END;
        """
        sql_update = "UPDATE key_value_store SET value = ? WHERE id = (SELECT key_value_store_id FROM tmp LIMIT 1) AND (SELECT COUNT(*) FROM tmp) = 1;"
        sql_insert_inst = f"""
        INSERT OR IGNORE INTO key_instances (name, key_name_id)
            VALUES {insert_instances};
        """
        sql_insert_kvs = "INSERT INTO key_value_store (value) SELECT ? WHERE (SELECT COUNT(*) FROM tmp) = 0;"
        sql_insert_kgm = f"""
        INSERT OR IGNORE INTO key_group_members (key_value_store_id, key_instance_id)
            VALUES {insert_members};
        """
        sqls_post = """
        DROP TABLE tmp;
        DROP TABLE kvs_last_inserted_rowid;
        DROP TRIGGER kvs_insert;
        """
        cursor = self._conn.cursor()
        cursor.executescript(sqls_prep)
        cursor.execute(sql_update, (_v,))
        cursor.execute(sql_insert_inst)
        cursor.execute(sql_insert_kvs, (_v,))
        cursor.execute(sql_insert_kgm)
        cursor.executescript(sqls_post)
        self._conn.commit()
        cursor.close()

        return None

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

        # Delete from key_value_store
        sql = f"DELETE FROM key_value_store WHERE id = {key_group_id}"
        cursor = self._conn.cursor()
        cursor.execute(sql)
        self._conn.commit()
        cursor.close()

        # Delete from key_group_members
        sql = f"DELETE FROM key_group_members WHERE key_value_store_id = {key_group_id}"
        cursor = self._conn.cursor()
        cursor.execute(sql)
        self._conn.commit()
        cursor.close()
        return None

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
              name TEXT,
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
