import os
from abc import ABC
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from labram.configs.types import PathStr, PathList, ensure_args_str


class CloudPath(ABC):
    SEP: PathStr = '/'
    HOST: PathStr = 'URL'
    URL_PREFIX: PathStr = 'URL_PREFIX://'

    @classmethod
    @ensure_args_str
    def with_prefix(cls, url: PathStr) -> bool:
        return url.startswith(cls.URL_PREFIX)

    @classmethod
    @ensure_args_str
    def is_url(cls, url: PathStr) -> bool:
        return cls.with_prefix(url)

    @classmethod
    @ensure_args_str
    def parts(cls, key: PathStr) -> PathList:
        return key.split(cls.SEP)

    @classmethod
    @ensure_args_str
    def join(cls, *keys: PathStr) -> PathStr:
        return cls.SEP.join(keys)

    @classmethod
    @ensure_args_str
    def host(cls, url_path: PathStr) -> str:
        return urlparse(url_path).scheme

    @classmethod
    @ensure_args_str
    def join_url_local(cls, url_path: PathStr, local_path: PathStr) -> PathStr:
        out_bucket_name, out_key = cls.parse_url(url_path)
        out_key = cls.join(out_key, local_path)
        out_url = cls.make_url(out_bucket_name, out_key)
        return out_url

    @classmethod
    @ensure_args_str
    def relative_path(cls, key: PathStr, base_key: PathStr) -> PathStr:
        len_path_base = len(cls.parts(base_key))
        rel_path_parts = cls.parts(key)[len_path_base:]
        rel_path = cls.join(*rel_path_parts)
        return rel_path

    @classmethod
    def os_path(cls, key: PathStr) -> PathStr:
        path_parts = cls.parts(key)
        return os.path.join(*path_parts)

    @classmethod
    @ensure_args_str
    def basename(cls, key: PathStr) -> PathStr:
        return cls.parts(key)[-1]

    @classmethod
    @ensure_args_str
    def suffix(cls, url: PathStr) -> PathStr:
        return Path(cls.basename(url)).suffix

    @classmethod
    @ensure_args_str
    def filename(cls, url: PathStr) -> PathStr:
        return Path(cls.basename(url)).stem

    @classmethod
    def dir_name(cls, key: PathStr) -> PathStr:
        if key[-1] == cls.SEP:
            dir_name = key[:-1]
        else:
            path_parts = cls.parts(key)
            dir_name = cls.join(*path_parts[:-1]) if len(path_parts) > 1 else path_parts[0]
        return dir_name

    @classmethod
    @ensure_args_str
    def split_head(cls, key: PathStr) -> (PathStr, PathStr):
        split = key.split(cls.SEP, maxsplit=1)
        head = split[0]
        tail = split[1] if len(split) > 1 else ''
        return head, tail

    @classmethod
    @ensure_args_str
    def from_os_path_to_url(cls, os_path: PathStr) -> PathStr:
        return cls.join(*Path(os_path).parts)

    @classmethod
    @ensure_args_str
    def parse_url(cls, url: PathStr) -> (PathStr, PathStr):
        """
        Split a URL to bucket name and key
        :param url: URL to parse
        :return: bucket, key
        """
        assert isinstance(url, str), f'url={url} must be a string'
        assert cls.is_url(url), f'url must start with prefix {cls.URL_PREFIX}, url={url}'

        bucket_name = urlparse(url).hostname

        key = url[len(cls.URL_PREFIX) + len(bucket_name) + 1:]
        return bucket_name, key

    @classmethod
    @ensure_args_str
    def make_url(cls, bucket: PathStr, key: Optional[PathStr] = None) -> PathStr:
        """
        Make a URL from bucket name and key
        """
        url_path = bucket
        if not cls.with_prefix(url_path):
            url_path = cls.URL_PREFIX + url_path
        if key is not None:
            url_path = url_path + cls.SEP + key
        return url_path

    @classmethod
    @ensure_args_str
    def norm_url(cls, url: PathStr) -> PathStr:
        norm_url = url
        if cls.is_url(norm_url):
            norm_url = norm_url[len(cls.URL_PREFIX):]

        norm_url = norm_url.replace('\\', '/')
        norm_url = norm_url.replace('//', '/')

        if norm_url.endswith('/'):
            norm_url = norm_url[:-1]

        norm_url = f'{cls.URL_PREFIX}{norm_url}'

        return norm_url
