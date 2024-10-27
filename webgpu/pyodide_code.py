
from .gpu import init_webgpu
from .utils import *
from .mesh import *
import js
import base64

gpu = None
mesh_object = None
render_function = None


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


async def reload(*args, **kwargs):
    print("reload")
    cleanup()
    reload_package("webgpu")
    from webgpu.main import main

    await main()

async def draw_cf(canvas_name, render_data):
    global gpu, mesh_object, render_function
    gpu = await init_webgpu(js.document.getElementById(canvas_name))
    device = Device(gpu.device)
    edges = base64.b64decode(render_data.edges)
    trigs = base64.b64decode(render_data.trigs)
    cf_vals = base64.b64decode(render_data.trig_function_values)
    edge_buffer = device.create_buffer(js.Uint8Array.new(edges),
                                       js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST)
    trigs_buffer = device.create_buffer(js.Uint8Array.new(trigs),
                                        js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST)
    cf_data_buffer = device.create_buffer(js.Uint8Array.new(cf_vals),
                                          js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST)
    mesh_object = CFRenderObject(gpu, {"edges": edge_buffer, "trigs": trigs_buffer,
                                   "trig_function_values" : cf_data_buffer }, render_data.n_trigs)
    wireframe_object = WireFrameRenderer(gpu, {"edges": edge_buffer, "trigs": trigs_buffer}, render_data.n_trigs)

    # move mesh to center and scale it
    for i in [0, 5, 10]:
        gpu.uniforms.mat[i] = 1.8

    gpu.uniforms.mat[15] = 1.0

    # translate to center
    gpu.uniforms.mat[12] = -0.5 * 1.8
    gpu.uniforms.mat[13] = -0.5 * 1.8

    t_last = 0
    fps = 0
    frame_counter = 0

    def render(time):
        nonlocal t_last, fps, frame_counter
        dt = time - t_last
        t_last = time
        frame_counter += 1
        print(f"frame time {dt:.2f} ms")
        gpu.uniforms.update_buffer()
        gpu.uniforms.update_buffer()

        command_encoder = gpu.device.createCommandEncoder()

        mesh_object.render(command_encoder)
        wireframe_object.render(command_encoder, loadOp="load")

        gpu.device.queue.submit([command_encoder.finish()])
        if frame_counter < 20:
            js.requestAnimationFrame(render_function)
    render_function = create_proxy(render)
    gpu.input_handler.render_function = render_function
    render_function.request_id = js.requestAnimationFrame(render_function)
    

async def draw_mesh(canvas_name, render_data):
    global gpu, mesh_object, render_function
    gpu = await init_webgpu(js.document.getElementById(canvas_name))
    device = Device(gpu.device)
    edges = base64.b64decode(render_data.edges)
    trigs = base64.b64decode(render_data.trigs)
    edge_buffer = device.create_buffer(js.Uint8Array.new(edges),
                                       js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST)
    trigs_buffer = device.create_buffer(js.Uint8Array.new(trigs),
                                        js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST)
    mesh_object = MeshRenderObject(gpu, {"edges": edge_buffer, "trigs": trigs_buffer}, render_data.n_trigs)
    wireframe_object = WireFrameRenderer(gpu, {"edges": edge_buffer, "trigs": trigs_buffer}, render_data.n_trigs)

    # move mesh to center and scale it
    for i in [0, 5, 10]:
        gpu.uniforms.mat[i] = 1.8

    gpu.uniforms.mat[15] = 1.0

    # translate to center
    gpu.uniforms.mat[12] = -0.5 * 1.8
    gpu.uniforms.mat[13] = -0.5 * 1.8

    t_last = 0
    fps = 0
    frame_counter = 0

    def render(time):
        nonlocal t_last, fps, frame_counter
        dt = time - t_last
        t_last = time
        frame_counter += 1
        print(f"frame time {dt:.2f} ms")
        gpu.uniforms.update_buffer()
        gpu.uniforms.update_buffer()

        command_encoder = gpu.device.createCommandEncoder()

        mesh_object.render(command_encoder)
        wireframe_object.render(command_encoder, loadOp="load")

        gpu.device.queue.submit([command_encoder.finish()])
        if frame_counter < 20:
            js.requestAnimationFrame(render_function)
    render_function = create_proxy(render)
    gpu.input_handler.render_function = render_function
    render_function.request_id = js.requestAnimationFrame(render_function)
