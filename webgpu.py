import asyncio
import js
from js import document, Float32Array, Int32Array
import pyodide.ffi
import sys

frame_counter = 0


def to_js(value):
    return pyodide.ffi.to_js(value, dict_converter=js.Object.fromEntries)


def abort():
    js.alert("WebGPU is not supported")
    sys.exit(1)


async def non_blocking_sleep(milliseconds):
    future = asyncio.Future()

    # Define a callback to resolve the future
    def resolve_future():
        future.set_result(None)

    # Use setTimeout to call the resolve function after the delay
    js.setTimeout(pyodide.ffi.create_proxy(resolve_future), milliseconds)

    # Await the future, allowing other tasks to run while waiting
    await future


def generate_data():
    if 1:
        from netgen.occ import unit_square

        m = unit_square.GenerateMesh(maxh=0.05)

        vertices = []
        for p in m.Points():
            for i in range(3):
                vertices.append(p[i])
            vertices.append(0)

        trigs = []
        edges = []
        for t in m.Elements2D():
            for i in range(3):
                trigs.append(t.vertices[i].nr - 1)
                edges.append(t.vertices[i].nr - 1)
                edges.append(t.vertices[(i + 1) % 3].nr - 1)
            trigs.append(0)
    else:
        vertices = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        trigs = [0, 1, 2, 0]
        edges = [0, 1, 1, 2, 2, 0]

    vertex_array = Float32Array.new(vertices)
    trigs_array = Int32Array.new(trigs)
    edges_array = Int32Array.new(edges)

    return vertex_array, edges_array, trigs_array


_render_function = None

_position = [0, 0]
_is_moving = False


def on_mousedown(ev):
    global _is_moving
    _is_moving = True


def on_mouseup(ev):
    global _is_moving
    _is_moving = False


def on_mousemove(ev):
    if _is_moving:
        _position[0] += ev.movementX
        _position[1] -= ev.movementY
        js.requestAnimationFrame(_render_function)


async def main(canvas=None, shader_url="./shader.wgsl"):

    if not js.navigator.gpu:
        abort()

    adapter = await js.navigator.gpu.requestAdapter()
    if not adapter:
        abort()

    device = await adapter.requestDevice()
    format = js.navigator.gpu.getPreferredCanvasFormat()

    if canvas is None:
        canvas = document.getElementById("canvas")

    context = canvas.getContext("webgpu")
    context.configure(
        to_js(
            {
                "device": device,
                "format": format,
                "alphaMode": "premultiplied",
            }
        )
    )

    canvas.addEventListener("mousedown", pyodide.ffi.create_proxy(on_mousedown))
    canvas.addEventListener("mouseup", pyodide.ffi.create_proxy(on_mouseup))
    canvas.addEventListener("mousemove", pyodide.ffi.create_proxy(on_mousemove))

    # Create the uniform buffer
    uniform_buffer = device.createBuffer(
        to_js(
            {
                "size": 2 * 4,
                "usage": js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST,
            }
        )
    )

    bindGroupLayout = device.createBindGroupLayout(
        to_js(
            {
                "entries": [
                    {
                        "binding": 0,
                        "visibility": js.GPUShaderStage.VERTEX,
                        "buffer": {"type": "uniform"},
                    },
                    {
                        "binding": 1,
                        "visibility": js.GPUShaderStage.VERTEX,
                        "buffer": {"type": "read-only-storage"},
                    },
                    {
                        "binding": 2,
                        "visibility": js.GPUShaderStage.VERTEX,
                        "buffer": {"type": "read-only-storage"},
                    },
                    {
                        "binding": 3,
                        "visibility": js.GPUShaderStage.VERTEX,
                        "buffer": {"type": "read-only-storage"},
                    },
                ],
            }
        )
    )

    vertices, edges, trigs = generate_data()

    vertex_buffer = device.createBuffer(
        to_js(
            {
                "size": vertices.length * 4,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )

    edges_buffer = device.createBuffer(
        to_js(
            {
                "size": edges.length * 4,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )

    trig_buffer = device.createBuffer(
        to_js(
            {
                "size": trigs.length * 4,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )

    device.queue.writeBuffer(vertex_buffer, 0, vertices)
    device.queue.writeBuffer(edges_buffer, 0, edges)
    device.queue.writeBuffer(trig_buffer, 0, trigs)

    # Create the bind group for the uniforms
    uniform_bind_group = device.createBindGroup(
        to_js(
            {
                "layout": bindGroupLayout,
                "entries": [
                    {"binding": 0, "resource": {"buffer": uniform_buffer}},
                    {"binding": 1, "resource": {"buffer": vertex_buffer}},
                    {"binding": 2, "resource": {"buffer": edges_buffer}},
                    {"binding": 3, "resource": {"buffer": trig_buffer}},
                ],
            }
        )
    )

    pipelineLayout = device.createPipelineLayout(
        to_js({"bindGroupLayouts": [bindGroupLayout]})
    )

    # Create the render pipeline
    shader_code = await (await js.fetch(shader_url)).text()
    render_pipeline_edges = device.createRenderPipeline(
        to_js(
            {
                "layout": pipelineLayout,
                "vertex": {
                    "module": device.createShaderModule(to_js({"code": shader_code})),
                    "entryPoint": "mainVertexEdge",
                },
                "fragment": {
                    "module": device.createShaderModule(to_js({"code": shader_code})),
                    "entryPoint": "mainFragmentEdge",
                    "targets": [{"format": format}],
                },
                "primitive": {"topology": "line-list"},
                "depthStencil": {
                    "format": "depth24plus",
                    "depthWriteEnabled": True,
                    "depthCompare": "less-equal",
                },
            }
        )
    )

    render_pipeline_trigs = device.createRenderPipeline(
        to_js(
            {
                "layout": pipelineLayout,
                "vertex": {
                    "module": device.createShaderModule(to_js({"code": shader_code})),
                    "entryPoint": "mainVertexTrig",
                },
                "fragment": {
                    "module": device.createShaderModule(to_js({"code": shader_code})),
                    "entryPoint": "mainFragmentTrig",
                    "targets": [{"format": format}],
                },
                "primitive": {"topology": "triangle-list"},
                "depthStencil": {
                    "format": "depth24plus",
                    "depthWriteEnabled": True,
                    "depthCompare": "less-equal",
                },
            }
        )
    )

    depthTexture = device.createTexture(
        to_js(
            {
                "size": [canvas.width, canvas.height, 1],
                "format": "depth24plus",
                "usage": js.GPUTextureUsage.RENDER_ATTACHMENT,
            }
        )
    )

    uniforms = Float32Array.new(2)

    async def update(time):
        global frame_counter
        # print("rendering image", frame_counter)
        frame_counter += 1
        uniforms[0] = _position[0] / canvas.width
        uniforms[1] = _position[1] / canvas.height
        device.queue.writeBuffer(uniform_buffer, 0, uniforms)

        command_encoder = device.createCommandEncoder()

        render_pass_encoder = command_encoder.beginRenderPass(
            to_js(
                {
                    "colorAttachments": [
                        {
                            "view": context.getCurrentTexture().createView(),
                            "clearValue": {"r": 1, "g": 1, "b": 1, "a": 1},
                            "loadOp": "clear",
                            "storeOp": "store",
                        }
                    ],
                    "depthStencilAttachment": {
                        "view": depthTexture.createView(
                            to_js({"format": "depth24plus", "aspect": "all"})
                        ),
                        "depthLoadOp": "clear",
                        "depthStoreOp": "store",
                        "depthClearValue": 1.0,
                    },
                },
            )
        )
        render_pass_encoder.setViewport(0, 0, canvas.width, canvas.height, 0.0, 0.9999)

        render_pass_encoder.setPipeline(render_pipeline_edges)
        render_pass_encoder.setBindGroup(0, uniform_bind_group)
        render_pass_encoder.draw(edges.length, 1, 0, 0)
        render_pass_encoder.setPipeline(render_pipeline_trigs)
        render_pass_encoder.setBindGroup(0, uniform_bind_group)
        render_pass_encoder.draw(trigs.length, 1, 0, 0)
        render_pass_encoder.end()

        device.queue.submit([command_encoder.finish()])
        # await non_blocking_sleep(1000/60)
        # js.requestAnimationFrame(pyodide.ffi.create_proxy(update))

    global _render_function
    _render_function = pyodide.ffi.create_proxy(update)

    js.requestAnimationFrame(_render_function)


main
