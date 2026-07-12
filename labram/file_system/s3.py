import os
import tempfile
import warnings
from abc import ABCMeta
from typing import Callable, Optional, Any
import boto3
import s3fs

from labram.file_system.os_filesystem import OsPath
from labram.file_system.cloud_filesystem import CloudPath
from labram.utils.types import PathStr, PathList, ensure_args_path


class S3Path(CloudPath):
    SEP = '/'
    HOST = 's3'
    URL_PREFIX = 's3://'


s3fs_metaclass = type(s3fs.S3FileSystem)


class MetaCombinedS3FS(s3fs_metaclass, ABCMeta):
    pass


class S3FS(s3fs.S3FileSystem, S3Path, metaclass=MetaCombinedS3FS):
    DEFAULT_REGION = 'eu-west-1'
    # Lazy-init boto3 client on first access — instantiating at class-body time
    # would require AWS credentials just to import this module.
    _BOTO3_CLIENT = None

    @classmethod
    def boto3_client(cls):
        if cls._BOTO3_CLIENT is None:
            cls._BOTO3_CLIENT = boto3.client('s3')
        return cls._BOTO3_CLIENT

    def __init__(self, *, key=None, secret=None, token=None):
        # `asynchronous` is intentionally not exposed here: the codebase performs
        # no async I/O, and older s3fs releases (<2021) reject the kwarg.
        s3_client_kwargs = {'region_name': S3FS.DEFAULT_REGION}
        s3fs.S3FileSystem.__init__(self,
                                   key=key,
                                   secret=secret,
                                   token=token,
                                   default_fill_cache=False,
                                   client_kwargs=s3_client_kwargs)

    @ensure_args_path
    def download_obj(self, url: PathStr, local_dir: PathStr, recursive: bool = False) -> PathStr:
        if not self.exists(url):
            raise FileExistsError(f'{url} is not valid file')
        if not os.path.isdir(os.path.dirname(local_dir)):
            raise FileExistsError(f'{local_dir} is not located in existing directory')

        if not url.endswith(self.SEP):
            url += self.SEP
        file_name = os.path.basename(url)
        local_file_path = os.path.join(local_dir, file_name)
        self.get(url, local_file_path, recursive=recursive)
        if not os.path.exists(local_file_path):
            raise FileExistsError(f"Download failed {url}->{local_file_path} local file not created")
        return local_file_path

    @ensure_args_path
    def upload_obj(self, local_path: PathStr, url: PathStr, recursive: bool = False) -> PathStr:
        if not os.path.exists(local_path):
            raise FileExistsError(f'{local_path} is not existing object')
        self.put(local_path, url, recursive=recursive)
        if not self.exists(url):
            raise FileExistsError(f"Upload failed {local_path}->{url} s3 file not created")
        return url

    def load_object(self, url: PathStr, loader: Callable) -> Any:
        if not self.isfile(url):
            raise FileExistsError(f'{url} is not valid file')

        suffix = self.suffix(url)
        with tempfile.NamedTemporaryFile(suffix=suffix) as local_temp_file:
            tmp_file_path = local_temp_file.name
            self.download(url, tmp_file_path)
            if not os.path.isfile(tmp_file_path):
                raise RuntimeError(f' Cannot load from: {url}')
            obj = loader(tmp_file_path)

        return obj

    def save_object(self, url: PathStr, saver: Callable[[Any, PathStr], Any], obj: object) -> PathStr:
        """
        Save an object 'obj' to s3 url via usage saver method
        Parameters
        ----------
        url - s3://bucket/path/to_my_dir/file.any
        saver - function that saves object to local file, saver(obj, local_path)
        obj - object to save

        Returns
        -------
         Path of saved file if succeeded otherwise None
        """
        with tempfile.NamedTemporaryFile(suffix=self.suffix(url)) as local_temp_file:
            tmp_file_path = local_temp_file.name
            saver(obj, tmp_file_path)
            self.upload(tmp_file_path, url)
        if not self.isfile(url):
            raise RuntimeError(f'Cannot save an object in s3 url :{url}')
        return url

    @ensure_args_path
    def list_obj(self, url: PathStr) -> PathList:
        if not self.isdir(url):
            raise FileExistsError(f'{url} is not valid dir')
        return list(map(self.make_url, self.ls(url)))

    @ensure_args_path
    def get_dir_size(self, path: PathStr) -> float:
        if not self.exists(path) or not self.isdir(path):
            warnings.warn(f's3 dir {path} does not exist or is not a directory')
            return 0

        bucket_name, prefix = self.parse_url(path)
        paginator = self.boto3_client().get_paginator('list_objects_v2')
        total_size = 0
        # Iterate through all objects in the specified S3 directory
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    total_size += obj['Size']
        return total_size

    def get_size(self, path: PathStr) -> Optional[float]:
        if self.exists(path):
            if self.isfile(path):
                return self.size(path)
            elif self.isdir(path):
                return self.get_dir_size(path)
        warnings.warn(f'Not valid s3 url: {path} object type or does not exist')
        return None

    def cp_objs(self, path1: PathStr, path2: PathStr,recursive: bool = True) -> None:
        if os.path.exists(path1) and self.is_url(path2):
            self.upload(path1, path2, recursive=recursive)
        elif self.exists(path1) and OsPath.is_url(path2):
            self.download(path1, path2, recursive=recursive)
        elif self.is_url(path1) and self.is_url(path2):
            self.cp(path1, path2, recursive=recursive)
        else:
            raise ValueError(f'path1={path1} path2={path2} are not valid paths')

    def exists(self, path: PathStr, **kwargs) -> bool:
        if self.is_url(path):
            return s3fs.S3FileSystem.exists(self, path, **kwargs)
        else:
            return False

    def isfile(self, path: PathStr) -> bool:
        if self.exists(path):
            return s3fs.S3FileSystem.isfile(self, path)
        else:
            return False