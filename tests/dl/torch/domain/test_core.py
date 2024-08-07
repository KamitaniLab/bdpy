"""Tests for bdpy.dl.torch.domain.core."""

import unittest
from bdpy.dl.torch.domain import core as core_module


class DummyAddDomain(core_module.Domain):
    def send(self, num):
        return num + 1
    
    def receive(self, num):
        return num - 1


class DummyDoubleDomain(core_module.Domain):
    def send(self, num):
        return num * 2
    
    def receive(self, num):
        return num // 2


class DummyUpperCaseDomain(core_module.Domain):
    def send(self, text):
        return text.upper()
    
    def receive(self, value):
        return value.lower()


class TestDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.core.Domain."""
    def setUp(self):
        self.domain = DummyAddDomain()
        self.original_space_num = 0
        self.internal_space_num = 1

    def test_instantiation(self):
        """Test instantiation."""
        self.assertRaises(TypeError, core_module.Domain)

    def test_send(self):
        """test send"""
        self.assertEqual(self.domain.send(self.original_space_num), self.internal_space_num)

    def test_receive(self):
        """test receive"""
        self.assertEqual(self.domain.receive(self.internal_space_num), self.original_space_num)

    def test_invertibility(self):
        input_candidates = [-1, 0, 1, 0.5]
        for x in input_candidates:
            assert x == self.domain.send(self.domain.receive(x))
            assert x == self.domain.receive(self.domain.send(x))


class TestInternalDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.core.InternalDomain."""
    def setUp(self):
        self.domain = core_module.InternalDomain()
        self.num = 1

    def test_send(self):
        """test send"""
        self.assertEqual(self.domain.send(self.num), self.num)

    def test_receive(self):
        """test receive"""
        self.assertEqual(self.domain.receive(self.num), self.num)

    def test_invertibility(self):
        input_candidates = [-1, 0, 1, 0.5]
        for x in input_candidates:
            assert x == self.domain.send(self.domain.receive(x))
            assert x == self.domain.receive(self.domain.send(x))


class TestIrreversibleDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.core.IrreversibleDomain."""
    def setUp(self):
        self.domain = core_module.IrreversibleDomain()
        self.num = 1

    def test_send(self):
        """test send"""
        self.assertEqual(self.domain.send(self.num), self.num)

    def test_receive(self):
        """test receive"""
        self.assertEqual(self.domain.receive(self.num), self.num)


class TestComposedDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.core.ComposedDomain."""
    def setUp(self):
        self.composed_domain = core_module.ComposedDomain([
            DummyDoubleDomain(),
            DummyAddDomain(),
        ])
        self.original_space_num = 0
        self.internal_space_num = 2

    def test_send(self):
        """test send"""
        self.assertEqual(self.composed_domain.send(self.original_space_num), self.internal_space_num)

    def test_receive(self):
        """test receive"""
        self.assertEqual(self.composed_domain.receive(self.internal_space_num), self.original_space_num)


class TestKeyValueDomain(unittest.TestCase):
    """Tests for bdpy.dl.torch.domain.core.KeyValueDomain."""
    def setUp(self):
        self.key_value_domain = core_module.KeyValueDomain({
            "name": DummyUpperCaseDomain(),
            "age": DummyDoubleDomain()
        })
        self.original_space_data = {"name": "alice", "age": 30}
        self.internal_space_data = {"name": "ALICE", "age": 60}

    def test_send(self):
        """test send"""
        self.assertEqual(self.key_value_domain.send(self.original_space_data), self.internal_space_data)

    def test_receive(self):
        """test receive"""
        self.assertEqual(self.key_value_domain.receive(self.internal_space_data), self.original_space_data)


if __name__ == "__main__":
    #unittest.main()
    composed_domain = core_module.ComposedDomain([
        DummyDoubleDomain(),
        DummyAddDomain(),
    ])
    print(composed_domain.receive(-1))
    print(composed_domain.send(-2))
