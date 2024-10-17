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
    if input_handler is not None:
        input_handler.unregister_callbacks()
    del mesh_object
    del input_handler
    del gpu


async def reload():
    print("reload")
    cleanup()
    import glob
    import importlib
    import os

    import webgpu
    import webgpu.colormap
    import webgpu.gpu
    import webgpu.main
    import webgpu.mesh
    import webgpu.utils

    dirname = os.path.dirname(__file__)
    for filename in glob.glob(os.path.join(dirname, "*.py")):
        if filename.endswith("__init__.py"):
            continue
        module_name = os.path.basename(filename)[:-3]
        webgpu.__dict__[module_name] = importlib.reload(webgpu.__dict__[module_name])
    webgpu = importlib.reload(webgpu)
    await webgpu.main.main()
