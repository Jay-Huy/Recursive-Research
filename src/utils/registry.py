from typing import Dict, Any, Type

class Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, Type] = {}

    def register(self, name: str):
        def wrap(obj: Type):
            assert name not in self._obj_map, f"An object named '{name}' was already registered in '{self._name}' registry!"
            self._obj_map[name] = obj
            return obj
        return wrap

    def build(self, name: str, **kwargs) -> Any:
        assert name in self._obj_map, f"'{name}' is not registered in '{self._name}' registry!"
        return self._obj_map[name](**kwargs)

MODEL_REGISTRY = Registry("MODEL")
DATASET_REGISTRY = Registry("DATASET")
METRIC_REGISTRY = Registry("METRIC")
