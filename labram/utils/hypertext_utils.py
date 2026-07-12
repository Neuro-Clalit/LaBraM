import os.path
import jsonpickle
import numpy as np
import json
import yaml
from typing import Any

from labram.file_system import FileSystem


class NumpyEncoder(json.JSONEncoder):
    """
    A JSONEncoder capable of converting numpy types to simple python builtin types.
    """

    # pylint: disable=method-hidden
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)

        return json.JSONEncoder.default(self, obj)

def from_data_file(file_url: str) -> Any:
    if file_url.endswith('.json'):
        return from_json(file_url)
    elif file_url.endswith('.yaml'):
        return from_yaml(file_url)
    else:
        raise TypeError(f'Non-supported format of file: {file_url }. '
                         f'Supported formats are JSON or YAML.')


def to_data_file(obj: Any, fil_url: str, indent: int = 4, encoder: json.JSONEncoder = NumpyEncoder) -> str:
    if fil_url.endswith('.json'):
        return to_json(obj, fil_url, indent, encoder)
    elif fil_url.endswith('.yaml'):
        return to_yaml(obj,fil_url)
    else:
        raise TypeError(f'Non-supported format of file: {fil_url}. '
                        f'Supported formats are JSON or YAML.')

def from_json(json_url: str) -> Any:
    return FileSystem().load_object(json_url, from_json_local)

def to_json(obj: Any, json_file: str, indent: int = 4, encoder: json.JSONEncoder = NumpyEncoder) -> str:
    # saver(obj, file_url)
    return FileSystem().save_object(json_file,
                                    saver= lambda obj_, file_path_: to_json_local(obj_, file_path_, indent, encoder),
                                    obj=obj)

def from_yaml(yaml_url: str) -> Any:
    return FileSystem().load_object(yaml_url, from_yaml_local)

def to_yaml(obj: object, yaml_url: str) -> str:
    # YAML files round-trip through safe_load, which rejects 'python/object:'
    # tags. Pre-convert ConfigBase (or any object exposing as_dict()) to a plain
    # nested dict so the file stays portable.
    payload = obj.as_dict(recursive=True) if hasattr(obj, 'as_dict') else obj
    return FileSystem().save_object(yaml_url,
                                    saver=lambda obj_, file_path: to_yaml_local(obj_, file_path),
                                    obj=payload)


def from_json_local(json_file: str) -> Any:
    if not os.path.isfile(json_file):
        raise FileExistsError(f'Not exist json_file: {json_file}')
    with open(json_file, 'r') as f:
        # keys=True opts in to jsonpickle 5.0's future default (preserves non-string
        # dict keys); aligns encode/decode and silences the deprecation warning.
        return jsonpickle.decode(f.read(), keys=True)

def to_json_local(obj: Any, json_file: str, indent: int = 4, encoder: json.JSONEncoder = NumpyEncoder) -> str:
    json_str = jsonpickle.encode(obj, keys=True)
    # load the encoded JSON so we can save it with in a pretty
    # format with indentations (jsonpickle does not support this)
    json_dict = json.loads(json_str)
    with open(json_file, 'w') as f:
        json.dump(json_dict, f, indent=indent, cls=encoder)
    if not os.path.isfile(json_file):
        raise RuntimeError(f'Cannot save to json file: {json_file}')
    return json_file


def from_yaml_local(yaml_url: str) -> Any:
    with open(yaml_url, 'r') as f:
        obj = yaml.safe_load(f)
    return obj

def to_yaml_local(obj: Any, output_path: str) -> str:
    with open(output_path, 'w') as f:
        yaml.dump(obj, f)
    if not os.path.isfile(output_path):
        raise RuntimeError(f'Cannot save to yaml file: {output_path}')
    return output_path


