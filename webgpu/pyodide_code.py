
from .gpu import init_webgpu
from .utils import *
from .mesh import *
import js
import base64

gpu = None
mesh_object = None
render_function = None

async def draw_mesh(canvas_name, render_data):
    global gpu, mesh_object, render_function
    print("received data = ", render_data)
    print("methods = ", dir(render_data))
    gpu = await init_webgpu(js.document.getElementById(canvas_name))
    device = Device(gpu.device)
    edges = base64.b64decode(render_data.edges)
    trigs = base64.b64decode(render_data.trigs)
    edge_buffer = device.create_buffer(js.Uint8Array.new(edges),
                                       js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST)
    trigs_buffer = device.create_buffer(js.Uint8Array.new(trigs),
                                        js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST)
    mesh_object = MeshRenderObject(gpu, {"edges": edge_buffer, "trigs": trigs_buffer}, render_data.n_trigs)

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

        gpu.device.queue.submit([command_encoder.finish()])
        if frame_counter < 20:
            js.requestAnimationFrame(render_function)
    render_function = create_proxy(render)
    gpu.input_handler.render_function = render_function
    render_function.request_id = js.requestAnimationFrame(render_function)
