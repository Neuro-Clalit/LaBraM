"""Tests for labram.configs.base_configs.ConfigBase and supporting type utilities.

Covers the public surface: load/save JSON+YAML, nested dataclass parsing,
type validation, update() with dotted keys, defaults, as_dict(), and __str__.
"""

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from labram.configs.base_configs import ConfigBase
from labram.configs.common.types import correct_type


# ----------------------------- fixture configs -----------------------------


@dataclass
class _Inner(ConfigBase):
    lr: float = 1e-3
    name: str = 'inner'


@dataclass
class _Outer(ConfigBase):
    epochs: int = 10
    tags: List[str] = field(default_factory=lambda: ['a', 'b'])
    inner: _Inner = field(default_factory=_Inner)
    optional_path: Optional[str] = None


@dataclass
class _OnlyPrimitives(ConfigBase):
    a: int = 1
    b: float = 2.5
    c: bool = True
    d: str = 'x'


# ----------------------------- helpers -----------------------------


@pytest.fixture
def tmp_file(tmp_path):
    """Yields a function that builds a unique filename under tmp_path."""
    counter = {'n': 0}

    def _make(suffix: str) -> str:
        counter['n'] += 1
        return str(tmp_path / f'cfg_{counter["n"]}{suffix}')

    return _make


# ----------------------------- defaults & dict ops -----------------------------


class TestDefaultsAndDict:
    def test_default_construction_populates_all_fields(self):
        outer = _Outer()
        assert outer.epochs == 10
        assert outer.tags == ['a', 'b']
        assert isinstance(outer.inner, _Inner)
        assert outer.inner.lr == 1e-3
        assert outer.optional_path is None

    def test_default_factory_returns_independent_instances(self):
        a = _Outer()
        b = _Outer()
        a.tags.append('mutated')
        a.inner.lr = 999.0
        assert b.tags == ['a', 'b']
        assert b.inner.lr == 1e-3

    def test_as_dict_recursive_flattens_nested_config(self):
        outer = _Outer(epochs=5, inner=_Inner(lr=0.01, name='foo'))
        d = outer.as_dict(recursive=True)
        assert d['epochs'] == 5
        assert d['inner'] == {'lr': 0.01, 'name': 'foo'}

    def test_as_dict_non_recursive_keeps_nested_config_instance(self):
        outer = _Outer()
        d = outer.as_dict(recursive=False)
        assert isinstance(d['inner'], _Inner)

    def test_getitem_dunder_returns_field_value(self):
        outer = _Outer(epochs=42)
        assert outer['epochs'] == 42

    def test_str_contains_classname_and_fields(self):
        outer = _Outer(epochs=7)
        s = str(outer)
        assert '_Outer' in s
        assert 'epochs' in s
        assert '7' in s


# ----------------------------- fields_names / exist / get_field -----------------------------


class TestFieldIntrospection:
    def test_fields_names_lists_all_dataclass_fields(self):
        names = set(_Outer.fields_names())
        assert names == {'epochs', 'tags', 'inner', 'optional_path'}

    def test_exist_true_for_declared_field(self):
        assert _Outer.exist('epochs') is True

    def test_exist_false_for_unknown_field(self):
        assert _Outer.exist('nonexistent') is False

    def test_get_field_returns_dataclass_field(self):
        f = _Outer.get_field('epochs')
        assert f.name == 'epochs'

    def test_is_dataclass_true_for_nested_config(self):
        assert _Outer.is_dataclass('inner') is True

    def test_is_dataclass_false_for_primitive(self):
        assert _Outer.is_dataclass('epochs') is False

    def test_call_default_factory_constructs_nested_default(self):
        instance = _Outer.call_default_factory('inner')
        assert isinstance(instance, _Inner)
        assert instance.lr == 1e-3

    def test_call_default_factory_returns_none_when_no_factory(self):
        # 'epochs' uses default=10, not default_factory — so factory call returns None
        assert _Outer.call_default_factory('epochs') is None


# ----------------------------- JSON round-trip -----------------------------


class TestJsonRoundTrip:
    def test_save_then_load_round_trips_primitive_config(self, tmp_file):
        cfg = _OnlyPrimitives(a=7, b=3.14, c=False, d='hello')
        path = tmp_file('.json')
        cfg.save_to(path)
        assert os.path.isfile(path)

        loaded = _OnlyPrimitives.load_from(path)
        assert loaded.a == 7
        assert loaded.b == 3.14
        assert loaded.c is False
        assert loaded.d == 'hello'

    def test_save_then_load_round_trips_nested_config(self, tmp_file):
        cfg = _Outer(epochs=42, tags=['x', 'y'], inner=_Inner(lr=0.05, name='deep'))
        path = tmp_file('.json')
        cfg.save_to(path)

        loaded = _Outer.load_from(path)
        assert loaded.epochs == 42
        assert loaded.tags == ['x', 'y']
        assert isinstance(loaded.inner, _Inner)
        assert loaded.inner.lr == 0.05
        assert loaded.inner.name == 'deep'


# ----------------------------- YAML round-trip -----------------------------


class TestYamlRoundTrip:
    def test_save_then_load_round_trips_primitive_config(self, tmp_file):
        cfg = _OnlyPrimitives(a=2, b=0.5, c=True, d='y')
        path = tmp_file('.yaml')
        cfg.save_to(path)
        assert os.path.isfile(path)

        loaded = _OnlyPrimitives.load_from(path)
        assert loaded.a == 2
        assert loaded.b == 0.5
        assert loaded.c is True
        assert loaded.d == 'y'

    def test_unsupported_extension_raises(self, tmp_file):
        cfg = _OnlyPrimitives()
        with pytest.raises(TypeError, match='Non-supported format'):
            cfg.save_to(tmp_file('.txt'))


# ----------------------------- load_from(dict) -----------------------------


class TestLoadFromDict:
    def test_load_from_dict_with_nested_dict_resolves_subconfig(self):
        data = {
            'epochs': 3,
            'tags': ['p', 'q'],
            'inner': {'lr': 0.02, 'name': 'sub'},
            'optional_path': '/tmp/out',
        }
        cfg = _Outer.load_from(data)
        assert cfg.epochs == 3
        assert cfg.tags == ['p', 'q']
        assert isinstance(cfg.inner, _Inner)
        assert cfg.inner.lr == 0.02
        assert cfg.inner.name == 'sub'
        assert cfg.optional_path == '/tmp/out'

    def test_load_from_dict_with_existing_subconfig_instance(self):
        data = {
            'epochs': 1,
            'tags': [],
            'inner': _Inner(lr=0.5, name='preset'),
            'optional_path': None,
        }
        cfg = _Outer.load_from(data)
        assert cfg.inner.lr == 0.5
        assert cfg.inner.name == 'preset'

    def test_load_from_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _Outer.load_from(42)  # type: ignore[arg-type]


# ----------------------------- load_config -----------------------------


class TestLoadConfig:
    def test_load_config_none_returns_defaults(self):
        cfg = _Outer.load_config(None)
        assert cfg.epochs == 10
        assert cfg.inner.lr == 1e-3

    def test_load_config_from_dict_applies_kwarg_overrides(self):
        # kwargs trigger backward-compat path that suppresses raise; just verify the override applies.
        cfg = _Outer.load_config(
            {'epochs': 4, 'tags': [], 'inner': {'lr': 1.0, 'name': 'n'}, 'optional_path': None},
            epochs=99,
        )
        assert cfg.epochs == 99

    def test_load_config_passes_through_existing_instance(self):
        seed = _Outer(epochs=11)
        out = _Outer.load_config(seed)
        assert out.epochs == 11

    def test_load_config_rejects_invalid_type(self):
        with pytest.raises(TypeError):
            _Outer.load_config(123)  # type: ignore[arg-type]


# ----------------------------- update() -----------------------------


class TestUpdate:
    def test_update_shallow_field_in_place(self):
        cfg = _Outer()
        cfg.update(epochs=200)
        assert cfg.epochs == 200

    def test_update_dotted_path_to_nested_field(self):
        cfg = _Outer()
        cfg.update(**{'inner.lr': 2e-4})
        assert cfg.inner.lr == 2e-4

    def test_update_unknown_key_raises_keyerror(self):
        cfg = _Outer()
        with pytest.raises(KeyError):
            cfg.update(**{'inner.does_not_exist.x': 1})


# ----------------------------- validate_type_fields -----------------------------


class TestValidateTypeFields:
    def test_passes_on_well_typed_instance(self):
        _Outer().validate_type_fields(raise_exception=True)  # no raise

    def test_raises_on_type_mismatch(self):
        cfg = _Outer()
        cfg.epochs = 'not-an-int'  # type: ignore[assignment]
        with pytest.raises(ValueError):
            cfg.validate_type_fields(raise_exception=True)

    def test_warns_when_raise_disabled(self):
        cfg = _Outer()
        cfg.epochs = 'not-an-int'  # type: ignore[assignment]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cfg.validate_type_fields(raise_exception=False)
        assert any('expected type' in str(w.message) for w in caught)


# ----------------------------- _cleanup_not_valid_values via load_from -----------------------------


class TestUnknownKeyCleanup:
    def test_unknown_top_level_key_is_dropped_with_warning(self):
        data = {
            'epochs': 2,
            'tags': [],
            'inner': {'lr': 0.1, 'name': 'n'},
            'optional_path': None,
            'bogus_key': 'should-be-removed',
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cfg = _Outer.load_from(data)
        assert cfg.epochs == 2
        assert any('bogus_key' in str(w.message) for w in caught)


# ----------------------------- correct_type helper -----------------------------


class TestCorrectType:
    @pytest.mark.parametrize("value,expected_type,ok", [
        (1, int, True),
        (1.5, float, True),
        ('s', str, True),
        (True, bool, True),
        (1, str, False),
        ('s', int, False),
        ([1, 2], List[int], True),
        ([1, 'x'], List[int], False),
        ([], List[int], True),
        (None, Optional[int], True),
        (5, Optional[int], True),
        ('s', Optional[int], False),
    ])
    def test_correct_type_matches_expectations(self, value, expected_type, ok):
        assert correct_type(value, expected_type) is ok
