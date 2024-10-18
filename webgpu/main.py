"""Main file for the webgpu example, creates a small 2d mesh and renders it using WebGPU"""

import js
import ngsolve as ngs
from netgen.occ import unit_square
from pyodide.ffi import create_proxy

from .gpu import init_webgpu
from .mesh import MeshRenderObject

gpu = None
mesh_object = None


async def main():
    global gpu, mesh_object

    gpu = await init_webgpu(js.document.getElementById("canvas"))

    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    order = 6
    gfu = ngs.GridFunction(ngs.H1(mesh, order=order))
    # gfu.Set(ngs.IfPos(ngs.x-0.8, 1, 0))
    N = 10
    gfu.Interpolate(ngs.sin(N * ngs.y) * ngs.sin(N * ngs.x))
    # gfu.Set(0.5*(ngs.x**order + ngs.y**order))
    # gfu.Set(ngs.y)
    mesh_object = MeshRenderObject(gpu)
    mesh_object.draw(gfu, mesh.Region(ngs.VOL), order=order)

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

        command_encoder = gpu.device.createCommandEncoder()

        render_pass_encoder = gpu.begin_render_pass(command_encoder)
        mesh_object.render(render_pass_encoder)
        render_pass_encoder.end()

        gpu.device.queue.submit([command_encoder.finish()])

    render_function = create_proxy(render)
    gpu.input_handler.render_function = render_function

    js.requestAnimationFrame(render_function)


def cleanup():
    print("cleanup")
    global gpu, mesh_object
    if "gpu" in globals():
        del gpu
    if "mesh_object" in globals():
        del mesh_object


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
