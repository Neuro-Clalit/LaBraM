import os
import tempfile
from typing import Optional, Callable, Any, Dict

from labram.file_system.cloud_filesystem import CloudPath
from labram.file_system.os_filesystem import OsPath
from labram.utils.types import RT, PathStr, PathList, ensure_args_path, args_type_check
from labram.file_system.s3 import S3FS, S3Path


class FileSystem(CloudPath):
    LOCAL_HOST = 'localhost'
    S3_HOST = S3FS.HOST
    HOSTS = [LOCAL_HOST, S3_HOST]
    host_paths: Dict[str, CloudPath] = {S3_HOST: S3Path, LOCAL_HOST: OsPath}

    def __init__(self, s3: Optional[S3FS] = None,
                 local_work_dir: Optional[PathStr] = None):
        # Defer S3FS construction until first S3 op — instantiating it eagerly
        # forces an s3fs/botocore session and breaks purely-local workflows
        # when AWS creds or a compatible s3fs version aren't available.
        self._s3_instance: Optional[S3FS] = s3
        self._temp_work_dir = tempfile.TemporaryDirectory(dir=local_work_dir)

    @property
    def _s3(self) -> S3FS:
        if self._s3_instance is None:
            self._s3_instance = S3FS()
        return self._s3_instance

    @classmethod
    @ensure_args_path
    def host_path(cls, path: PathStr) -> CloudPath:
        return cls.host_paths[cls.host_name(path)]

    @classmethod
    @ensure_args_path
    def host_name(cls, path: str) -> str:
        host = cls.host(path)
        if host == '':
            host = cls.LOCAL_HOST
        if host not in cls.HOSTS:
            raise ValueError(f'Not supported  host type of {path}')
        return host

    @classmethod
    @ensure_args_path
    def is_local(cls, path: PathStr) -> bool:
        # Accept absolute paths (starts with os.sep) AND relative paths (anything not S3).
        return not S3Path.is_url(path)

    @ensure_args_path
    def exists(self, path: PathStr) -> bool:
        if self.is_s3_url(path):
            return self._s3.exists(path)
        else:  # local — absolute or relative
            return os.path.exists(path)

    @args_type_check
    def rm(self, path: PathStr, recursive: bool = True) -> None:
        return self._map_host_fun(path,
                                  host_methods={self.LOCAL_HOST: lambda x: OsPath.rm(x, recursive=recursive),
                                                self.S3_HOST: lambda x: self._s3.rm(x, recursive=recursive)})


    @ensure_args_path
    def size(self, path: PathStr) -> float:
        return self._map_host_fun(path,
                                  host_methods={self.LOCAL_HOST: os.path.getsize,
                                                self.S3_HOST: self._s3.size})

    @ensure_args_path
    def get_dir_size(self, path: PathStr) -> float:
        return self._map_host_fun(path,
                                  host_methods={self.LOCAL_HOST: OsPath.get_dir_size,
                                                self.S3_HOST: self._s3.get_dir_size})

    @ensure_args_path
    def isfile(self, path: PathStr) -> bool:
        return self._map_host_fun(path,
                                  host_methods={self.LOCAL_HOST: OsPath.isfile,
                                                self.S3_HOST: self._s3.isfile},
                                  must_exist=False)

    @ensure_args_path
    def isdir(self, path: PathStr) -> bool:
        return self._map_host_fun(path,
                                  host_methods={self.LOCAL_HOST: os.path.isdir,
                                                self.S3_HOST: self._s3.isdir})

    def load_object(self, file_path: PathStr, loader: Callable[[PathStr], Any]) -> Any:
        """
        Load an object from remote or local filesystem
        Parameters
        ----------
        file_path - file path to load the object from a filesystem
        loader - a method to load the object, for example 'np.load' from saved in a local file

        Returns loaded object instance from remote or local filesystem
        -------

        """
        return self._map_host_fun(file_path,
                                  host_methods={self.LOCAL_HOST: loader,
                                                self.S3_HOST: lambda file_url: self._s3.load_object(file_url, loader)})

    def save_object(self,
                    file_url: PathStr,
                    saver: Callable[[Any, PathStr], PathStr],
                    obj: Any) -> PathStr:
        """
        Save an object 'obj' to remote or local filesystem via usage saver method
        Parameters
        ----------
        url - url to remote storage or local path
        saver - function that saves object to local file, saver(obj, local_path)
        obj - object to save

        Returns
        -------
         Path of saved file if succeeded otherwise None
        """
        if self.is_local(file_url):
            os.makedirs(os.path.dirname(file_url), exist_ok=True)
        out_url = self._map_host_fun(file_url,
                                     must_exist=False,
                                     host_methods={
                                         self.LOCAL_HOST: lambda path_: saver(obj, path_),
                                         self.S3_HOST: lambda url_: self._s3.save_object(url_, saver, obj)})

        if not self.isfile(out_url):
            raise RuntimeError(f'Cannot save an object in local path:{out_url}')
        return out_url

    def list_obj(self,
                 path: PathStr,
                 fun_filter: Optional[Callable[[PathStr], PathStr]] = None) -> PathList:
        dir_contains = self._map_host_fun(path,
                                          host_methods={self.LOCAL_HOST: OsPath.list_obj,
                                                        self.S3_HOST: self._s3.list_obj})

        if fun_filter:
            dir_contains = list(filter(fun_filter, dir_contains))
        return dir_contains

    def copy(self, path_src: PathStr, path_dst: PathStr, recursively: bool = False):
        if not self.exists(path_src):
            raise FileExistsError(f'path_src: {path_src}')
        if self.is_local(path_src) and self.is_local(path_dst):
            return OsPath.copy(path_src, path_dst, recursively=recursively)
        elif self.is_s3_url(path_src) or self.is_s3_url(path_dst):
            return self._s3.cp_objs(path_src,
                                    path_dst,
                                    recursive=recursively)
        else:
            raise ValueError(f'path1={path_src} path2={path_dst} are not valid paths')

    def get_local(self,
                  path: PathStr,
                  local_path: Optional[PathStr] = None,
                  recursive: bool = False) -> PathStr:
        # def get_s3()
        # local_path = local_path if local_path else self._temp_work_dir.name
        return self._map_host_fun(path,
                           host_methods={self.LOCAL_HOST: lambda url_: url_,
                                         self.S3_HOST: lambda url_: self._s3.download_obj(url_,
                                                                                 local_path,
                                                                                 recursive=recursive)})

        # return local_path

    def _map_host_fun(self,
                      path: str,
                      host_methods: Dict[str, Callable[[str], RT]],
                      must_exist: bool = True) -> RT:
        #  host_methods = {self.LOCAL_HOST: fun_local, self.S3_HOST: fun_s3}
        if not self.exists(path) and must_exist:
            raise RuntimeError(f'Not a valid path (does not exist): {path}')

        if set(host_methods.keys()) != set(self.HOSTS):
            raise RuntimeError(f'Missmatch in host_methods: {host_methods} and available: {self.HOSTS}')
        # Use is_s3_url directly so relative local paths (e.g. ./results) are handled correctly.
        host_name = self.S3_HOST if self.is_s3_url(path) else self.LOCAL_HOST
        return host_methods[host_name](path)

    @property
    def s3(self) -> S3FS:
        return self._s3

    @ensure_args_path
    def is_s3_url(self, path: PathStr) -> bool:
        # Use the classmethod so we don't lazily construct an S3FS just to check a prefix.
        return S3Path.is_url(path)

    @ensure_args_path
    def is_url(self, path: PathStr) -> bool:
        return self._map_host_fun(path,
                                  must_exist=False,
                                  host_methods={self.LOCAL_HOST: OsPath.is_url,
                                                self.S3_HOST: lambda url_: S3Path.is_url(url_)})
