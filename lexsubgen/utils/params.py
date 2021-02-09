import importlib
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from _jsonnet import evaluate_file

from lexsubgen.utils.register import logger


class Params(MutableMapping):
    DEFAULT_VALUE = object

    def __init__(self, params: Dict):
        """
        Objects of this class represents parameters dictionary.
        Using this class one could build other objects with build_from_params function.
        You may consume parameters with pop method and at the end check that all parameters
        were read. More precisely its a wrapper around ordinary `dict` which supports some
        auxiliary functionality.

        Args:
            params: dictionary representing parameters.
        """
        self.__dict__.update(params)

    @property
    def dict(self) -> Dict:
        """
        Get underlying parameters dictionary.

        Returns:
            parameters dictionary
        """
        return self.__dict__

    def get(
        self, key: str, default_value: object = DEFAULT_VALUE, _type: Optional = None
    ):
        """
        Implements functionality of `dict.get(key)` method but also check for the the type of
        returned value. If it's a dict then it will convert it into Params object. Also it supports
        conversion of return value to the specified standard type.

        Args:
            key: identifier of the object to get from the Params
            default_value: default value to return if there is no object with specified key
            _type: type to which the return object should be converted

        Returns:
            object from Params
        """
        if default_value is self.DEFAULT_VALUE:
            value = self.__dict__.get(key)
        else:
            value = self.__dict__.get(key, default_value)
        value = self._convert_value(value)
        if _type is not None:
            value = self._convert_type(value, _type)
        return value

    def pop(
        self, key: str, default_value: object = DEFAULT_VALUE, _type: Optional = None
    ):
        """
        Performs functionality of `dict.pop` method but additionally converts dict return object
        into Params objects. Also return values could be converted to the specified type. If there is no
        object with specified key then default value will be return if it's given.
        The object with specified key will be removed from Params.

        Args:
            key: identifier of the object to get from the Params
            default_value: default value to return if there is no object with specified key
            _type: type to which the return object should be converted

        Returns:
            object from Params
        """
        if default_value is self.DEFAULT_VALUE:
            value = self.__dict__.pop(key)
        else:
            value = self.__dict__.pop(key, default_value)
        value = self._convert_value(value)
        if _type is not None:
            value = self._convert_type(value, _type)
        return value

    def _convert_value(self, value):
        """
        Checks the type of the value.
        If it's a dict then converts it to Params object.
        If it's a list then it goes through the list and subsequently performs
        conversion of each element.

        Args:
            value: object to process

        Returns:
            converted object
        """
        if isinstance(value, dict):
            return Params(value)
        elif isinstance(value, list):
            value = [self._convert_value(item) for item in value]
        return value

    @staticmethod
    def _convert_type(value, _type):
        """
        Converts object to the given type.

        Args:
            value: object to be converted
            _type: type to which the object should be converted

        Returns:
            converted object
        """
        if value is None or value == "None":
            return None
        if _type is bool:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                if value == "false":
                    return False
                if value == "true":
                    return True
                raise TypeError("To convert to bool value should be bool or str.")
        return _type(value)

    def __getitem__(self, key):
        if key in self.__dict__:
            return self._convert_value(self.__dict__[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


def read_config(config_path: str, verbose: bool = False):
    """
    Builds config object from configuration file
    Args:
        config_path: path to a configuration file in json or jsonnet format.
        verbose: Bool flag for verbosity.
    Returns:
        Built config object
    """
    logger.info("Loading configuration file...")
    config_path = Path(config_path)
    if config_path.suffix == ".jsonnet":
        config = json.loads(evaluate_file(str(config_path)))
    elif config_path.suffix == ".json":
        with open(config_path, "r") as fp:
            config = json.load(fp)
    else:
        raise ValueError(
            "Configuration files should be provided in json or jsonnet format"
        )
    logger.info(f"Loaded configuration: {config}")
    return config


def build_from_config_path(
    config_path: str = None, config: Optional[Dict] = None
) -> Tuple[Any, Dict]:
    """
    Builds object from a configuration.

    Args:
        config_path: path to configuration file
        config: configuration represented as `dict`

    Returns:
        object that was build from parameters, object configuration
    """

    assert config_path is not None or config is not None
    if config_path is not None and config is None:
        config = read_config(config_path, verbose=True)

    params = Params(config)

    return build_from_params(params), config


def clsname2cls(clsname: str):
    import_path = "lexsubgen"
    if "." in clsname:
        module_path, clsname = clsname.rsplit(".", 1)
        import_path += "." + module_path
    # try:
    module = importlib.import_module(import_path)
    cls = getattr(module, clsname)
    # except Exception as e:
    #     raise ValueError(f"Failed to import '{clsname}'")
    return cls


def build_from_params(params: Params):
    """
    Builds object from parameters. Parameters must contain a 'class_name' field
    indicating where to import the class from.

    Args:
        params: parameters from that the object will be build.

    Returns:
        object that was build from parameters.
    """
    if isinstance(params, int):
        return params

    if "class_name" in params:
        cls_name = params.pop("class_name")
        cls = clsname2cls(cls_name)

        # init params acquisition
        kwargs = {}
        keys = list(params.keys())
        for key in keys:
            item_params = params.pop(key)
            if isinstance(item_params, Params):
                item = build_from_params(item_params)
            elif isinstance(item_params, list):
                item = [build_from_params(elem_params) for elem_params in item_params]
            else:
                item = item_params
            kwargs[key] = item
        return cls(**kwargs)
    return params
