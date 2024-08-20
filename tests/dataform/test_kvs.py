import os
import unittest
import tempfile
import sqlite3

import numpy as np

from bdpy.dataform import SQLite3KeyValueStore


class TestSQlite3KeyValueStore(unittest.TestCase):

    def _init_test_db(self, db_path):
        sqls = [
            # Create tables
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
            # Insert keys
            "INSERT OR IGNORE INTO key_names (name) VALUES ('layer')",
            "INSERT OR IGNORE INTO key_names (name) VALUES ('subject')",
            "INSERT OR IGNORE INTO key_names (name) VALUES ('roi')",
            "INSERT OR IGNORE INTO key_names (name) VALUES ('metric')",
        ]
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for sql in sqls:
            cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()

    def _dump_db(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM key_names")
        key_names = cursor.fetchall()
        cursor.execute("SELECT * FROM key_instances")
        key_instances = cursor.fetchall()
        cursor.execute("SELECT * FROM key_value_store")
        key_value_store = cursor.fetchall()
        cursor.execute("SELECT * FROM key_group_members")
        key_group_members = cursor.fetchall()
        cursor.close()
        conn.close()

        print("key_names:")
        print(key_names)
        print("key_instances:")
        print(key_instances)
        print("key_value_store:")
        print(key_value_store)
        print("key_group_members:")
        print(key_group_members)

    def test_initialize_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize DB with keys
            db_path = os.path.join(tmpdir, "test.db")
            keys = ["layer", "subject", "roi", "metric"]

            kvs = SQLite3KeyValueStore(db_path, keys=keys)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM key_names")
            key_names = cursor.fetchall()
            cursor.close()
            conn.close()

            self.assertEqual(key_names[0][0], "layer")
            self.assertEqual(key_names[1][0], "subject")
            self.assertEqual(key_names[2][0], "roi")
            self.assertEqual(key_names[3][0], "metric")

            # Initialize DB without keys
            db_nokey_path = os.path.join(tmpdir, "test_nokey.db")
            with self.assertRaises(ValueError):
                kvs_nokey = SQLite3KeyValueStore(db_nokey_path)

    def test_load_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_3304.db")
            self._init_test_db(db_path)

    def test_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_3304.db")
            self._init_test_db(db_path)

            kvs = SQLite3KeyValueStore(db_path)

            kvs.set(np.array([3.3, 0.4]), layer="conv1", subject="sub01", roi="V1", metric="accuracy")
            kvs.set(np.array([12, 34]), layer="conv1", subject="sub01", roi="V2", metric="accuracy")

            kvs.set(np.array(np.nan), layer="conv1", subject="sub02", roi="V1", metric="accuracy")

            with self.assertRaises(ValueError):
                kvs.set(np.array([3.3, 0.4]), layer="conv1", subject="sub01", roi="V1", metric="accuracy", invalid_key="invalid")

            with self.assertRaises(ValueError):
                kvs.set(np.array([3.3, 0.4]), layer="conv1", subject="sub01", roi="V1")

    def test_set_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_3304.db")
            self._init_test_db(db_path)

            kvs = SQLite3KeyValueStore(db_path)

            # Not found (None)
            val = kvs.get(layer="conv1", subject="sub_never_exsit", roi="V1", metric="accuracy")
            assert val is None

            # Found
            kvs.set(np.array([ 1,  2,  3,  4]), layer="conv1", subject="sub03", roi="LOC", metric="accuracy")
            kvs.set(np.array([ 5,  6,  7,  8]), layer="conv1", subject="sub03", roi="FFA", metric="accuracy")
            kvs.set(np.array([ 9, 10, 11, 12]), layer="conv1", subject="sub03", roi="PPA", metric="accuracy")
            val = kvs.get(layer="conv1", subject="sub03", roi="LOC", metric="accuracy")
            np.testing.assert_array_equal(val, np.array([ 1,  2,  3,  4]))
            val = kvs.get(layer="conv1", subject="sub03", roi="FFA", metric="accuracy")
            np.testing.assert_array_equal(val, np.array([ 5,  6,  7,  8]))
            val = kvs.get(layer="conv1", subject="sub03", roi="PPA", metric="accuracy")
            np.testing.assert_array_equal(val, np.array([ 9, 10, 11, 12]))

            # Found (empty array)
            kvs.set(np.array([]), layer="conv1", subject="sub04", roi="LOC", metric="accuracy")
            val = kvs.get(layer="conv1", subject="sub04", roi="LOC", metric="accuracy")
            np.testing.assert_array_equal(val, np.array([]))

            # Found (np.nan)
            kvs.set(np.array([np.nan]), layer="conv1", subject="sub04", roi="FFA", metric="accuracy")
            val = kvs.get(layer="conv1", subject="sub04", roi="FFA", metric="accuracy")
            np.testing.assert_array_equal(val, np.array([np.nan]))

    def test_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_3304.db")
            self._init_test_db(db_path)

            kvs = SQLite3KeyValueStore(db_path)

            kvs.set(np.array([ 1,  2,  3,  4]), layer="conv1", subject="sub03", roi="LOC", metric="accuracy")
            kvs.set(np.array([ 5,  6,  7,  8]), layer="conv1", subject="sub03", roi="FFA", metric="accuracy")
            kvs.set(np.array([np.nan]),         layer="conv1", subject="sub03", roi="PPA", metric="accuracy")

            val = kvs.get(layer="conv1", subject="sub03", roi="PPA", metric="accuracy")
            if np.array_equal(val, np.array([np.nan]), equal_nan=True):
                kvs.set(np.array([10, 20, 30, 40]), layer="conv1", subject="sub03", roi="PPA", metric="accuracy")
            val = kvs.get(layer="conv1", subject="sub03", roi="PPA", metric="accuracy")
            np.testing.assert_array_equal(val, np.array([10, 20, 30, 40]))

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_3304.db")
            self._init_test_db(db_path)

            kvs = SQLite3KeyValueStore(db_path)

            kvs.set(np.array([ 1,  2,  3,  4]), layer="conv1", subject="sub03", roi="LOC", metric="accuracy")
            kvs.set(np.array([ 5,  6,  7,  8]), layer="conv1", subject="sub03", roi="FFA", metric="accuracy")
            kvs.set(np.array([np.nan]),         layer="conv1", subject="sub03", roi="PPA", metric="accuracy")

            kvs.delete(layer="conv1", subject="sub03", roi="PPA", metric="accuracy")
            np.testing.assert_(~kvs.exists(layer="conv1", subject="sub03", roi="PPA", metric="accuracy"),
                               'AssertionError: Failed to delete the record.')

    def test_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_3304.db")
            self._init_test_db(db_path)

            kvs = SQLite3KeyValueStore(db_path)

            kvs.set(np.array([ 1,  2,  3,  4]), layer="conv1", subject="sub03", roi="LOC", metric="accuracy")
            kvs.set(np.array([ 5,  6,  7,  8]), layer="conv1", subject="sub03", roi="FFA", metric="accuracy")
            np.testing.assert_(kvs.lock(layer="conv1", subject="sub03", roi="PPA", metric="accuracy"),
                               'AssertionError: Failed to lock the specified condition.')
            np.testing.assert_(~kvs.lock(layer="conv1", subject="sub03", roi="PPA", metric="accuracy"),
                               'AssertionError: A condition that was already locked has been newly locked.')


if __name__ == "__main__":
    unittest.main()
