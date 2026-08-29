import os

def fix_relative_path(path, base):
    """Return path relative to base, if possible."""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path

def fix_absolute_path(path, base):
    """Return absolute path by joining base and relative path."""
    if not os.path.isabs(path):
        return os.path.abspath(os.path.join(base, path))
    return path
