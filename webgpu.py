import js
import pyodide.ffi
from pyodide.ffi import create_proxy
import sys
import ctypes as ct
import math


class ClippingPlane(ct.Structure):
    _fields_ = [("normal", ct.c_float * 3), ("dist", ct.c_float)]


class Complex(ct.Structure):
    _fields_ = [("re", ct.c_float), ("im", ct.c_float)]


class Colormap(ct.Structure):
    _fields_ = [("min", ct.c_float), ("max", ct.c_float)]


class Uniforms(ct.Structure):
    _fields_ = [
        ("mat", ct.c_float * 16),
        ("clipping_plane", ClippingPlane),
        ("colormap", Colormap),
        ("scaling", Complex),
        ("aspect", ct.c_float),
        ("eval_mode", ct.c_uint32),
        ("do_clipping", ct.c_uint32),
        ("padding", ct.c_uint32),
    ]


def to_js(value):
    return pyodide.ffi.to_js(value, dict_converter=js.Object.fromEntries)


def abort():
    js.alert("WebGPU is not supported")
    sys.exit(1)


def generate_data():
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

    vertex_array = js.Float32Array.new(vertices)
    trigs_array = js.Int32Array.new(trigs)
    edges_array = js.Int32Array.new(edges)

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


async def init_canvas(canvas):
    if not js.navigator.gpu:
        abort()

    if canvas is None:
        canvas = js.document.getElementById("canvas")

    # cloning and replacing the canvas removes all old event listeners
    new_canvas = canvas.cloneNode(True)
    canvas.parentNode.replaceChild(new_canvas, canvas)
    canvas = new_canvas
    del new_canvas

    canvas.addEventListener("mousedown", create_proxy(on_mousedown))
    canvas.addEventListener("mouseup", create_proxy(on_mouseup))
    canvas.addEventListener("mousemove", create_proxy(on_mousemove))

    return canvas


def create_colormap(device):
    n = 256
    data = js.Uint8Array.new(n * 4)

    for i in range(n):
        data[4 * i] = i
        data[4 * i + 1] = n - i - 1
        data[4 * i + 2] = 0
        data[4 * i + 3] = 255

    texture = device.createTexture(
        to_js(
            {
                "dimension": "1d",
                "size": [n, 1, 1],
                "format": "rgba8unorm",
                "usage": js.GPUTextureUsage.TEXTURE_BINDING
                | js.GPUTextureUsage.COPY_DST,
            }
        )
    )

    device.queue.writeTexture(
        to_js({"texture": texture}), data, to_js({"bytesPerRow": n * 4}), [n, 1, 1]
    )

    sampler = device.createSampler(
        to_js(
            {
                "magFilter": "linear",
                "minFilter": "linear",
                "addressModeU": "repeat",
                "addressModeV": "clamp-to-edge",
            }
        )
    )
    return texture, sampler


def create_buffers(device):
    uniform_buffer = device.createBuffer(
        to_js(
            {
                "size": len(bytes(Uniforms())),
                "usage": js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST,
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

    return uniform_buffer, vertex_buffer, edges_buffer, trig_buffer


async def main(canvas=None, shader_url="./shader.wgsl"):

    canvas = await init_canvas(canvas)

    adapter = await js.navigator.gpu.requestAdapter()
    if not adapter:
        abort()

    device = await adapter.requestDevice()
    format = js.navigator.gpu.getPreferredCanvasFormat()

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
    colormap_texture, colormap_sampler = create_colormap(device)

    bindGroupLayout = device.createBindGroupLayout(
        to_js(
            {
                "entries": [
                    {
                        "binding": 0,
                        "visibility": js.GPUShaderStage.VERTEX
                        | js.GPUShaderStage.FRAGMENT,
                        "buffer": {"type": "uniform"},
                    },
                    {
                        "binding": 1,
                        "visibility": js.GPUShaderStage.FRAGMENT,
                        "texture": {
                            "sampleType": "float",
                            "viewDimension": "1d",
                            "multisamples": False,
                        },
                    },
                    {
                        "binding": 2,
                        "visibility": js.GPUShaderStage.FRAGMENT,
                        "sampler": {"type": "filtering"},
                    },
                    {
                        "binding": 3,
                        "visibility": js.GPUShaderStage.VERTEX,
                        "buffer": {"type": "read-only-storage"},
                    },
                    {
                        "binding": 4,
                        "visibility": js.GPUShaderStage.VERTEX,
                        "buffer": {"type": "read-only-storage"},
                    },
                ],
            }
        )
    )

    uniform_buffer, vertex_buffer, edge_buffer, trig_buffer = create_buffers(device)

    edge_bind_group = device.createBindGroup(
        to_js(
            {
                "layout": bindGroupLayout,
                "entries": [
                    {"binding": 0, "resource": {"buffer": uniform_buffer}},
                    {"binding": 1, "resource": colormap_texture.createView()},
                    {"binding": 2, "resource": colormap_sampler},
                    {"binding": 3, "resource": {"buffer": vertex_buffer}},
                    {"binding": 4, "resource": {"buffer": edge_buffer}},
                ],
            }
        )
    )

    trig_bind_group = device.createBindGroup(
        to_js(
            {
                "layout": bindGroupLayout,
                "entries": [
                    {"binding": 0, "resource": {"buffer": uniform_buffer}},
                    {"binding": 1, "resource": colormap_texture.createView()},
                    {"binding": 2, "resource": colormap_sampler},
                    {"binding": 3, "resource": {"buffer": vertex_buffer}},
                    {"binding": 4, "resource": {"buffer": trig_buffer}},
                ],
            }
        )
    )

    pipelineLayout = device.createPipelineLayout(
        to_js({"bindGroupLayouts": [bindGroupLayout]})
    )

    shader_code = await (await js.fetch(shader_url)).text()
    shader_module = device.createShaderModule(to_js({"code": shader_code}))
    render_pipeline_edges = device.createRenderPipeline(
        to_js(
            {
                "layout": pipelineLayout,
                "vertex": {
                    "module": shader_module,
                    "entryPoint": "mainVertexEdge",
                },
                "fragment": {
                    "module": shader_module,
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
                    "module": shader_module,
                    "entryPoint": "mainVertexTrig",
                },
                "fragment": {
                    "module": shader_module,
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

    uniforms = Uniforms()
    uniforms.do_clipping = 1
    uniforms.clipping_plane.normal[0] = 1
    uniforms.clipping_plane.normal[1] = 0
    uniforms.clipping_plane.normal[2] = 0
    uniforms.clipping_plane.dist = 1
    uniforms.colormap.min = 0.0
    uniforms.colormap.max = 0.0
    uniforms.scaling.im = 0.0
    uniforms.scaling.re = 0.0
    uniforms.aspect = 0.0
    uniforms.eval_mode = 0

    for i in range(16):
        uniforms.mat[i] = 0.0

    for i in [0, 5, 10]:
        uniforms.mat[i] = 1.8

    uniforms.mat[15] = 1.0

    # translate to center
    uniforms.mat[12] = -0.5 * 1.8
    uniforms.mat[13] = -0.5 * 1.8

    async def update(time):
        uniforms.mat[12] += _position[0] / canvas.width
        uniforms.mat[13] += _position[1] / canvas.height
        _position[0] = 0
        _position[1] = 0
        # print("render", time)
        # uniforms.clipping_plane.dist = math.sin(time/300)* 0.5 - 0.5
        data = bytes(uniforms)
        buffer = js.Uint8Array.new(data)
        device.queue.writeBuffer(uniform_buffer, 0, buffer)

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
        render_pass_encoder.setViewport(0, 0, canvas.width, canvas.height, 0.0, 1.0)

        render_pass_encoder.setPipeline(render_pipeline_edges)
        render_pass_encoder.setBindGroup(0, edge_bind_group)
        render_pass_encoder.draw(len(edge_buffer) / 4, 1, 0, 0)

        render_pass_encoder.setPipeline(render_pipeline_trigs)
        render_pass_encoder.setBindGroup(0, trig_bind_group)
        render_pass_encoder.draw(len(trig_buffer) / 4, 1, 0, 0)

        render_pass_encoder.end()

        device.queue.submit([command_encoder.finish()])

    global _render_function
    _render_function = create_proxy(update)

    js.requestAnimationFrame(_render_function)


main
