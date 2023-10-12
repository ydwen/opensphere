from importlib import import_module

def build_from_args(args, module):
    """instantiate an class object from a config dict.
    Args:
        args (dict): Config dict. It should at least contain the key "type".
        module (string): module for import.
        package (string): for relative imports.
    Returns:
        object: The constructed object.
    """
    if not isinstance(args, dict):
        raise TypeError(f'args must be a dict, but got {type(args)}')
    if 'type' not in args:
        raise KeyError('Missing key "type" in `args`', args)

    obj_type = args.pop('type')
    if not isinstance(obj_type, str):
        raise TypeError(f'type must be a str, but got {type(obj_type)}')

    obj_cls = getattr(import_module(module), obj_type)

    return obj_cls(**args)
