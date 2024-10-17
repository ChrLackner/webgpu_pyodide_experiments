import js
from pyodide.ffi import create_proxy
from pyodide.ffi import to_js as _to_js


def to_js(value):
    return _to_js(value, dict_converter=js.Object.fromEntries)
