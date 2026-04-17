from typing import Any

import torch


def dump_config(config, writer):
    def _walk(section: dict[str, Any], param_name: str) -> None:
        for key, value in section.items():
            sub_param_name = f"{param_name}.{key}" if param_name != "" else key

            if isinstance(value, dict):
                _walk(value, sub_param_name)
            elif isinstance(value, list):
                _walk(dict(enumerate(value)), sub_param_name)
            else:
                if torch.is_tensor(value):
                    params[sub_param_name] = str(value.item())
                else:
                    params[sub_param_name] = str(value)

    params: dict[str, str] = {}
    _walk(config, param_name="")
    for key, value in params.items():
        writer.add_text(key, value)
