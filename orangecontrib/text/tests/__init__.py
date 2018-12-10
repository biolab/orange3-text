import unittest
import os


def suite(loader=unittest.TestLoader(), pattern='test*.py'):
    """Loads all project's tests."""
    dir_ = os.path.dirname(os.path.dirname(__file__))
    top_level = os.path.realpath(os.path.join(dir_, "..", ".."))
    all_tests = loader.discover(dir_, pattern, top_level_dir=top_level)
    return unittest.TestSuite(all_tests)
