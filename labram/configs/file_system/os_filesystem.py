import os
import shutil
from pathlib import Path

from labram.configs.file_system.cloud_filesystem import CloudPath
from labram.configs.common.types import ensure_args_path, PathStr, PathList


class OsPath(CloudPath):
    SEP = os.sep
    HOST = SEP
    URL_PREFIX = SEP

    @classmethod
    @ensure_args_path
    def list_obj(cls, path: PathStr) -> PathList:
        return list(map(str, Path(path).iterdir()))

    @classmethod
    @ensure_args_path
    def get_dir_size(cls, path: PathStr) -> float:
        if not os.path.isdir(path):
            raise FileNotFoundError(f'{path} is not a directory')
        total_size = 0
        for dir_path, dir_names, file_names in os.walk(path):
            for filename in file_names:
                file_path = os.path.join(dir_path, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size

    @classmethod
    @ensure_args_path
    def rm(cls, path: PathStr, recursively: bool = False) -> None:
        try:
            if recursively:
                shutil.rmtree(path)
                print(f"Successfully removed {path} recursively.")
            else:
                os.remove(path)
                print(f"Successfully removed {path}.")
        except IsADirectoryError:
            print(f"{path} is a directory. Use the 'recursively' flag to remove it.")
        except Exception as e:
            print(f"Error removing {path}: {e}")

    @classmethod
    @ensure_args_path
    def isfile(cls, path: PathStr) -> bool:
        if os.path.exists(path):
            return os.path.isfile(path)
        else:
            return False

    @classmethod
    @ensure_args_path
    def copy(cls, path_src: PathStr, path_dist: PathStr ,recursively: bool = False) -> None:
        try:
            if recursively:
                shutil.copytree(path_src, path_dist, dirs_exist_ok=True)
                print(f"Successfully copy {path_src}->{path_dist} recursively.")
            else:
                shutil.copyfile(path_src, path_dist)
                # print(f"Successfully copy {path}.")
        except IsADirectoryError:
            print(f"{path_src} is a directory. Use the 'recursively' flag to remove it.")
        except Exception as e:
            print(f"Error copy {path_src}->{path_dist}: {e}")
