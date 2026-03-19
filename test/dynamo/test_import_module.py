# Owner(s): ["module: dynamo"]
import unittest
import torch

class ImportUserModuleTypeTests(unittest.TestCase):    
    def test_import_user_define_module(self):
        """
        testcase for https://github.com/pytorch/pytorch/issues/177682
        Bad import result for types.ModuleType subclass in sys.modules
        """
        import types
        import sys
        class _ConfigModule(types.ModuleType):
            x = 1

        _ConfigModule.__module__ = __name__
        sys.modules["my_config"] = _ConfigModule("my_config")

        @torch.compile(fullgraph=True, backend="eager")
        def f():
            import my_config  # noqa: F401
            return torch.tensor(1)
        try:
            f()
        except Exception:
            self.assertTrue(False, msg="compile error should not be raised")
        else:
            self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
