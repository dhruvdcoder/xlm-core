from importlib import import_module
from functools import lru_cache


@lru_cache(maxsize=None)
def get_function(path: str):
    """Quicky way to import a function from a string.

    Args:
        path (str): The path to the function.

    Returns:
        The function.

    Example:
        >>> get_function("xlm.datamodule.ids_to_example_fn")
        <function xlm.datamodule.ids_to_example_fn>

    Limitations:
        1. Must include at least one “.”

        `get_function("foo")        # → ValueError: not enough values to unpack`
        You need a dot to split module vs. attribute.

        2. No nested‐attribute support
            It only splits on the last dot. If you wrote "pkg.mod.Class.method", you’d get pkg.mod.Class as the module and then .method on that, which gives you an unbound function, not a bound method of an instance.

        3. Import failures
            If the module name is wrong or missing from your PYTHONPATH, you’ll get an ImportError or ModuleNotFoundError at lookup time.

        4. Attribute errors
            If the attribute isn’t on the module (typo, private name, etc.), you’ll see an AttributeError. You might want to catch and rewrap these with a friendlier message.

        5. Dynamic‐reload blind spot
            Because it caches the object, if you later hot-reload the module or monkey-patch it, get_function will keep returning the old reference. (You can clear the cache with get_function.cache_clear() if you really need to.)

        6. Built-ins & extensions
            For truly built-in callables (e.g. len, open) you’d have to use "builtins.len" or "io.open"—plain "len" won’t work.
    """
    module_name, fn_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, fn_name)
