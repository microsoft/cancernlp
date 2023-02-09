"""
Path structures supporting 'lazy' evaluation.

This allows scripts to modify arbitrary path components, and have these changes
picked up by preconfigured sub-paths.

For instance, I may preconfigure a path:

# give some options for how to find base directory
BASE_DIR = LazyPath([LazyEnvVar["BASE_DIR"], "/mnt/base"])
# now preconfigure some branching structure
SUB_DIR_A = LazyPath(BASE_DIR, "some/path/to/A")
SUB_DIR_B = LazyPath(BASE_DIR, "some/path/to/B")
FILE_A_1 = LazyPath(SUB_DIR_A, "file1.txt")
FILE_A_2 = LazyPath(SUB_DIR_A, "file2.txt")
FILE_B_1 = LazyPath(SUB_DIR_B, "file1.txt")
FILE_B_2 = LazyPath(SUB_DIR_B, "file2.txt")


# Later, in some end-user script, we may want to:
# change base location of all the data:
BASE_DIR.set_abs_path("/home/myuser/data")
# or maybe just of some particular subdirectory:
SUB_DIR_A.set_abs_path("/different/version/of/A")

# either way, this is transparent to any code that uses any of the FILE_?_?'s.
"""
import os
from typing import List, Optional, Union


class LazyEnvVar:
    def __init__(self, var: str):
        """
        Lazy evaluation of an environment variable.  The value is not obtained until
        accessed by requestion code (i.e. in the final runtime environment)

        :param var: name of the environment variable
        """
        self.var = var

    def get(self):
        """Resolve the environment variable and return the value"""
        return os.environ.get(self.var)

    def __str__(self):
        return f"${self.var}={self.get()}"


# paths can either be an explicit string or an environment variable
PathLike = Union[str, LazyEnvVar, "LazyPath"]
MultiPathLike = Union[PathLike, List[PathLike]]


class LazyPath:
    def __init__(
        self,
        paths_or_parent: Union["LazyPath", MultiPathLike],
        paths: Optional[MultiPathLike] = None,
    ):
        """
        Class representing a file system path, lazily evaluated when needed.

        This allows scripts to modify arbitrary path components, and have these changes
        picked up by preconfigured sub-paths.

        Construction options:
        # a base path
        LazyPath("/explicit/path")
        # a base path determined by environment variable
        LazyPath(a_lazy_env_var)
        # a base path determined by environment variable if available, else explicit
        path
        LazyPath([a_lazy_env_var, "/explicit/path"])
        # a path built on top of a parent LazyPath
        LazyPath(parent_lazy_path, "sub_path")
        # a path built on top of a parent LazyPath, determined by environment variable
        LazyPath(parent_lazy_path, a_lazy_env_var)
        """
        self.parent = None
        if isinstance(paths_or_parent, LazyPath):
            self.parent = paths_or_parent
        else:
            if paths is not None:
                raise ValueError(
                    "Error, if two arguments are given, first must be parent path"
                )
            paths = paths_or_parent
        if not isinstance(paths, list):
            paths = [paths]
        self.paths = paths

    def get_path(self, require_valid=False):
        try:
            return self._get_path(require_valid=require_valid)
        except FileNotFoundError as fnfe:
            raise FileNotFoundError(
                "Error while trying to construct path "
                f"{self.construct_human_readable_path()}: {str(fnfe)}"
            )

    def _get_path(self, require_valid=False):
        """
        Resolve the path.  If require_valid is True, raise an error if no path can
        be resolved that exists on the current file system.
        """
        path_pieces = []
        if self.parent:
            path_pieces.append(self.parent._get_path(require_valid=require_valid))

        for path in self.paths:
            if isinstance(path, LazyEnvVar):
                path = path.get()
                if path is None:
                    continue
            if isinstance(path, LazyPath):
                path = path.get_path(require_valid=require_valid)
            if not require_valid or os.path.exists(os.path.join(*path_pieces, path)):
                break
        else:
            raise FileNotFoundError(self.construct_human_readable_path())
        path_pieces.append(path)
        return os.path.join(*path_pieces)

    def get_valid_path(self):
        """Resolve the path, raising an error if the path does not exist."""
        return self.get_path(require_valid=True)

    def set_abs_path(self, path):
        """
        Override this path with an absolute path, ignoring parents and affecting
        children.
        """
        self.parent = None
        self.paths = [path]

    def set_rel_path(self, path):
        """
        Override this path with a path relative to the existing parent.
        """
        self.paths = [path]

    def construct_human_readable_path(self) -> str:
        path = ""
        if self.parent:
            path = self.parent.construct_human_readable_path() + "/"
        # cast each element to str in case it's a LazyEnvVar
        path += "[" + "|".join([str(p) for p in self.paths]) + "]"
        return path

    def __str__(self):
        return self.construct_human_readable_path()


class lazy_path_class_param(object):
    """
    Adaptor to create a class parameter from a LazyPath that looks like a property,
    but that calls get_path() when accessed.

    example:
    >>> class A:
    >>>     the_path = lazy_path_class_param(LazyPath(parent_lazy_path, "somedir"))
    >>>
    >>> # A.the_path is now a string path (not a LazyPath):
    >>> os.path.exists(A.the_path)
    True
    """

    def __init__(self, lazy_path: LazyPath, require_valid: bool = True):
        self.lazy_path = lazy_path
        self.require_valid = require_valid

    def __get__(self, obj, owner):
        return self.lazy_path.get_path(require_valid=self.require_valid)
