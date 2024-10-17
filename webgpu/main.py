import sys

import js
from pyodide.ffi import create_proxy

from .gpu import WebGPU
from .input_handler import InputHandler
from .mesh import MeshRenderObject

gpu = None
input_handler = None
mesh_object = None


async def main(canvas=None):
    global gpu, input_handler, mesh_object

    if not js.navigator.gpu:
        js.alert("WebGPU is not supported")
        sys.exit(1)

    canvas = canvas or js.document.getElementById("canvas")
    adapter = await js.navigator.gpu.requestAdapter()

    if not adapter:
        js.alert("WebGPU is not supported")
        sys.exit(1)

    device = await adapter.requestDevice()

    gpu = WebGPU(device, canvas)
    input_handler = InputHandler(gpu)

    from netgen.occ import unit_square

    mesh = unit_square.GenerateMesh(maxh=0.3)
    mesh_object = MeshRenderObject(mesh, gpu)

    # move mesh to center and scale it
    for i in [0, 5, 10]:
        gpu.uniforms.mat[i] = 1.8

    gpu.uniforms.mat[15] = 1.0

    # translate to center
    gpu.uniforms.mat[12] = -0.5 * 1.8
    gpu.uniforms.mat[13] = -0.5 * 1.8

    def render(time):
        # this is the render function, it's called for every frame

        # copy camera position etc. to GPU
        gpu.uniforms.update_buffer()

        command_encoder = device.createCommandEncoder()

        render_pass_encoder = gpu.begin_render_pass(command_encoder)
        mesh_object.draw(render_pass_encoder)
        render_pass_encoder.end()

        device.queue.submit([command_encoder.finish()])

    gpu.render_function = create_proxy(render)

    js.requestAnimationFrame(gpu.render_function)


def cleanup():
    print("cleanup")
    global gpu, input_handler, mesh_object
    if "input_handler" in globals():
        if input_handler is not None:
            input_handler.unregister_callbacks()
        del mesh_object
        del input_handler
        del gpu


def reload_package(package_name):
    """Reload python package and all submodules (searches in modules for references to other submodules)"""
    import importlib
    import os
    import types

    package = importlib.import_module(package_name)
    assert hasattr(package, "__package__")
    file_name = package.__file__
    package_dir = os.path.dirname(file_name) + os.sep
    reloaded_modules = {file_name: package}

    def reload_recursive(module):
        module = importlib.reload(module)

        for var in vars(module).values():
            if isinstance(var, types.ModuleType):
                file_name = getattr(var, "__file__", None)
                if file_name is not None and file_name.startswith(package_dir):
                    if file_name not in reloaded_modules:
                        reloaded_modules[file_name] = reload_recursive(var)

        return module

    reload_recursive(package)
    return reloaded_modules


async def reload():
    print("reload")
    cleanup()
    reload_package("webgpu")
    from webgpu.main import main

    await main()
