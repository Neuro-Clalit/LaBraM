import json
import warnings
from abc import ABC
import inspect
from dataclasses import dataclass, Field, fields, MISSING
from typing import ClassVar, List, Optional, Tuple, Union, AnyStr, Any, get_type_hints

import labram.configs.common.hypertext_utils as hyper_utils
from labram.configs.file_system import FileSystem
from labram.configs.common.types import OPT_STR, DATA_DICT, correct_type

DATA_OBJ = Union[DATA_DICT, 'ConfigBase']


@dataclass
class ConfigBase(ABC):
    # Lazy shared FileSystem — instantiated on first access to avoid an eager
    # S3 / boto3 connection at import time.
    _FS_INSTANCE: ClassVar[Optional[FileSystem]] = None

    @classmethod
    def _get_fs(cls) -> FileSystem:
        if ConfigBase._FS_INSTANCE is None:
            ConfigBase._FS_INSTANCE = FileSystem()
        return ConfigBase._FS_INSTANCE

    # Keep FS_ as a convenience alias so existing code using self.FS_ still works.
    @property
    def FS_(self) -> FileSystem:
        return self._get_fs()

    def __str__(self) -> AnyStr:
        s = f'\n[{self.__class__.__name__} -----'
        for key, value in sorted(self.__dict__.items()):
            s += f'\n o {key:45} | {value}'
        s += f'\n ----- {self.__class__.__name__}]'
        return s

    def as_dict(self, recursive: bool = True) -> DATA_DICT:
        dict_obj = self.__dict__.copy()
        if recursive:
            for key, value in dict_obj.items():
                if isinstance(value, ConfigBase):
                    dict_obj[key] = value.as_dict(recursive)
        return dict_obj

    def __getitem__(self, key: AnyStr) -> Any:
        return self.__dict__[key]

    @classmethod
    def get(cls, key: AnyStr) -> Any:
        return cls.__dict__.get(key)

    @classmethod
    def _from_dict(cls, dict_obj: DATA_DICT, raise_exception: bool = True) -> 'ConfigBase':
        # TODO: raise_exception backward compatibility must raise_exception = TRUE always
        missmatch_fields = list(
            set(dict_obj.keys()).symmetric_difference(set(cls.fields_names()))
        )
        if len(missmatch_fields) > 0:
            err_msg = f'{cls.__name__}: The following fields are missing: \n{missmatch_fields}'
            if raise_exception:
                raise ValueError(err_msg)
            else:
                warnings.warn(err_msg)
        for field in cls.get_fields():
            try:
                field_name = field.name
                if  cls.is_dataclass(field_name) and isinstance(dict_obj[field_name], dict):
                    dict_obj[field_name] = field.type.from_dict(dict_obj[field_name])
            except TypeError as err:
                # TODO backward compatibility: must raise exception
                warnings.warn(f'Field: {field_name}: {str(err)}')
        config = cls(**dict_obj)
        config.validate_type_fields(raise_exception)
        return config

    def save_to(self,
                file_path: str,
                indent: int = 4,
                encoder: json.JSONEncoder = hyper_utils.NumpyEncoder) -> str:
        # Dispatch by file suffix (.json / .yaml) — symmetric with load_from.
        return hyper_utils.to_data_file(obj=self,
                                        fil_url=file_path,
                                        indent=indent,
                                        encoder=encoder)

    @classmethod
    def fields_names(cls) -> List[str]:
        # dataclasses.fields() excludes ClassVar/InitVar pseudo-fields, which
        # __dataclass_fields__.keys() would incorrectly include (e.g. _FS_INSTANCE).
        return [f.name for f in fields(cls)]

    @classmethod
    def get_field(cls, name: str) -> Field:
        return cls.__dataclass_fields__[name]

    @classmethod
    def call_default_factory(cls, name: str) -> Optional[Any]:
        field_ = cls.get_field(name)
        if not field_.default_factory == MISSING:
            return field_.default_factory()
        return None

    @classmethod
    def get_fields(cls) -> Tuple[Field, ...]:
        return fields(cls)

    @classmethod
    def load_from(cls, data_obj: DATA_OBJ, raise_exception: bool = True) -> 'ConfigBase':
        if isinstance(data_obj, str):
            data_obj = hyper_utils.from_data_file(data_obj)
        data_obj = cls._parse_data_obj(data_obj)
        data_obj.validate_type_fields(raise_exception)
        return data_obj

    @classmethod
    def _parse_data_obj(cls, data_obj: DATA_OBJ, raise_exception: bool = True) -> 'ConfigBase':
        # TODO: raise_exception backward compatibility must raise_exception = TRUE always
        if isinstance(data_obj, ConfigBase):
            data_obj = data_obj.__dict__
        elif not isinstance(data_obj, dict):
            raise TypeError(f'data_obj must dict of data class type: {type(data_obj)}')
        data_obj = cls._cleanup_not_valid_values(data_obj)
        for name, val in sorted(data_obj.items()):
            field_instance = cls.call_default_factory(name)
            if isinstance(val, ConfigBase):
                data_obj[name] = val._parse_data_obj(val, raise_exception=raise_exception)
            elif isinstance(val, dict) and isinstance(field_instance, ConfigBase):
                data_obj[name] = field_instance._parse_data_obj(val, raise_exception=raise_exception)
        data_obj = cls._from_dict(data_obj, raise_exception=raise_exception)
        return data_obj

    @classmethod
    def _cleanup_not_valid_values(cls, obj_dict: DATA_DICT) -> DATA_DICT:
        non_valid_keys = [key for key in obj_dict.keys() if not cls.exist(key)]
        if len(non_valid_keys) > 0:
            warnings.warn(f'Not valid name: {non_valid_keys}, will be removed from config dict')
            [obj_dict.pop(key_to_remove, None) for key_to_remove in non_valid_keys]
        return obj_dict

    @classmethod
    def load_config(cls,
                    config_obj: Union[str, dict, 'ConfigBase', None],
                    raise_exception: bool = True,
                    **kwargs) -> 'ConfigBase':
        # TODO added for backward compatibility,must be True always
        raise_exception = (len(kwargs) == 0) and raise_exception
        if config_obj is None:
            config = cls()
        elif isinstance(config_obj, (str, dict)):
            config = cls.load_from(config_obj, raise_exception)
        elif isinstance(config_obj, cls):
            config = cls._parse_data_obj(config_obj)
        else:
            raise TypeError(f'config_obj not valid type: {type(config_obj)}')
        config = config.update(**kwargs)
        config.validate_type_fields()
        return config

    def update(self, **kwargs) -> 'ConfigBase':
        """
        Updates the values of given configuration elements
        :param kwargs: The names and values of the configuration elements to update.
        :return: self
        """

        try:
            for key, value in kwargs.items():
                c = self.__dict__
                *sub_keys, final_key = key.split('.')
                for sub_key in sub_keys:
                    c = c[sub_key].__dict__
                c[final_key] = value
        except KeyError as e:
            raise KeyError(f'Non-valid update input {kwargs} , error message: {e}')

        return self

    def validate_type_fields(self, raise_exception: bool = True) -> None:
        """ Validates that each field in a dataclass instance conforms to its type annotation. """
        # TODO: raise_exception backward compatibility
        #  must raise_exception = TRUE always
        for field_ in self.get_fields():
            value_ = getattr(self, field_.name)
            if not correct_type(value_, field_.type):
                err_message = (f"Object: '{type(self)}', "
                               f"Field '{field_.name}', "
                               f"expected type {field_.type}, "
                               f"with value: {value_}, "
                               f"but got {type(value_).__name__}.")
                if raise_exception:
                    raise ValueError(str(err_message))
                else:
                    warnings.warn(f'{str(err_message)} \n '
                                  f'backward compatibility: must be raise_exception')
                    continue

            if isinstance(value_, ConfigBase):
                value_.validate_type_fields(raise_exception)

    @classmethod
    def is_field_class(cls, name: str) -> bool:
        field = cls.get_field(name)
        return inspect.isclass(field.type)

    @classmethod
    def is_dataclass(cls, field_name: str) -> bool:
        # Use get_type_hints to resolve string annotations (PEP 563 / from __future__)
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}
        type_ = hints.get(field_name)
        return type_ is not None and inspect.isclass(type_) and issubclass(type_, ConfigBase)

    @classmethod
    def exist(cls, field_name: str) -> bool:
        return field_name in cls.fields_names()

