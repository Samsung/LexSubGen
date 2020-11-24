from copy import deepcopy
from itertools import product
from typing import Dict, List

from lexsubgen.utils.params import Params, clsname2cls


class Grid(Params):
    HYPERPARAM_NAMES = ["Hyperparam", "LinspaceHyperparam", "LogspaceHyperparam"]

    def __init__(self, params: Dict):
        super(Grid, self).__init__(params=params)
        self.hyperparams = self.get_hyperparam_values()
        self.param_names, self.param_values = list(
            zip(*[(param["name"], param["values"]) for param in self.hyperparams])
        )

    def get_hyperparam_values(self) -> List[Dict]:
        params = deepcopy(self.dict)
        hyperparam_values = list()
        self.dfs(params, hyperparam_values)
        return hyperparam_values

    def __len__(self):
        num_grid_dots = 1
        for values in self.param_values:
            num_grid_dots *= len(values)
        return num_grid_dots

    def __iter__(self):
        for grid_dot in product(*self.param_values):
            params = deepcopy(self.dict)
            params.pop("hyperparams")
            params.pop("param_names")
            params.pop("param_values")
            grid_dot = list(grid_dot)
            hypers = deepcopy(grid_dot)
            self.dfs(params, grid_dot, is_fill=True)
            yield hypers, Params(params)

    def dfs(self, item, accumulated: List, is_fill: bool = False) -> List:
        hyperparams = []
        if isinstance(item, dict) and item.get("class_name", None):
            if is_fill:
                classname = item["class_name"]
            else:
                classname = item.pop("class_name")

            if classname in self.HYPERPARAM_NAMES:
                cls = clsname2cls(classname)
                hyperparam = cls(**item)
                accumulated.append(
                    {"name": hyperparam.name, "values": hyperparam.values}
                )
            else:
                keys = list(item.keys())
                for key in keys:
                    if is_fill:
                        _item = item[key]
                        if isinstance(_item, dict) and _item.get("class_name", None):
                            _classname = _item["class_name"]
                            if _classname in self.HYPERPARAM_NAMES:
                                item[key] = accumulated[0]
                                accumulated.pop(0)
                                continue
                    else:
                        _item = item.pop(key)
                    prev_size = len(accumulated)
                    self.dfs(_item, accumulated, is_fill)
                    if not is_fill and prev_size != len(accumulated):
                        last_added_param = accumulated[-1]
                        name = last_added_param["name"]
                        if name is None:
                            last_added_param["name"] = key
        elif isinstance(item, list):
            for _item in item:
                self.dfs(_item, accumulated, is_fill)
        return hyperparams
