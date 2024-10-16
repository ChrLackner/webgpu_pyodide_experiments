import ctypes as ct
import sys

import js
import pyodide.ffi
from pyodide.ffi import create_proxy


class ClippingPlaneUniform(ct.Structure):
    _fields_ = [("normal", ct.c_float * 3), ("dist", ct.c_float)]


class ComplexUniform(ct.Structure):
    _fields_ = [("re", ct.c_float), ("im", ct.c_float)]


class ColormapUniform(ct.Structure):
    _fields_ = [("min", ct.c_float), ("max", ct.c_float)]


class Uniforms(ct.Structure):
    _fields_ = [
        ("mat", ct.c_float * 16),
        ("clipping_plane", ClippingPlaneUniform),
        ("colormap", ColormapUniform),
        ("scaling", ComplexUniform),
        ("aspect", ct.c_float),
        ("eval_mode", ct.c_uint32),
        ("do_clipping", ct.c_uint32),
        ("padding", ct.c_uint32),
    ]

    def __init__(self):
        self.do_clipping = 1
        self.clipping_plane.normal[0] = 1
        self.clipping_plane.normal[1] = 0
        self.clipping_plane.normal[2] = 0
        self.clipping_plane.dist = 1
        self.colormap.min = 0.0
        self.colormap.max = 0.0
        self.scaling.im = 0.0
        self.scaling.re = 0.0
        self.aspect = 0.0
        self.eval_mode = 0

        for i in range(16):
            self.mat[i] = 0.0


class InputHandler:
    def __init__(self, gpu):
        self.gpu = gpu
        self._is_moving = False

        canvas = self.gpu.canvas
        canvas.addEventListener("mousedown", create_proxy(self.on_mousedown))
        canvas.addEventListener("mouseup", create_proxy(self.on_mouseup))
        canvas.addEventListener("mousemove", create_proxy(self.on_mousemove))

    def on_mousedown(self, _):
        self._is_moving = True

    def on_mouseup(self, _):
        global _is_moving
        self._is_moving = False

    def on_mousemove(self, ev):
        if self._is_moving:
            self.gpu.uniforms.mat[12] += ev.movementX / self.gpu.canvas.width * 1.8
            self.gpu.uniforms.mat[13] -= ev.movementY / self.gpu.canvas.height * 1.8
            js.requestAnimationFrame(self.gpu.render_function)


class WebGPU:
    """WebGPU management class, handles "global" state, like device, canvas, colormap and uniforms"""

    def __init__(self, device, canvas):
        self.render_function = None
        self.device = device
        self.format = js.navigator.gpu.getPreferredCanvasFormat()
        self.canvas = canvas

        self.uniforms = Uniforms()

        uniforms_size = len(bytes(self.uniforms))
        if uniforms_size % 16:
            raise ValueError(
                f"Uniforms size must be multiple of 16, current size: {uniforms_size}"
            )

        self.context = canvas.getContext("webgpu")
        self.context.configure(
            to_js(
                {
                    "device": device,
                    "format": self.format,
                    "alphaMode": "premultiplied",
                }
            )
        )
        self.uniform_buffer = device.createBuffer(
            to_js(
                {
                    "size": len(bytes(self.uniforms)),
                    "usage": js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST,
                }
            )
        )
        self.colormap_texture, self.colormap_sampler = create_colormap(device)
        self.depth_format = "depth24plus"
        self.depth_stencil = {
            "format": self.depth_format,
            "depthWriteEnabled": True,
            "depthCompare": "less",
        }

        self.depth_texture = device.createTexture(
            to_js(
                {
                    "size": [canvas.width, canvas.height, 1],
                    "format": self.depth_format,
                    "usage": js.GPUTextureUsage.RENDER_ATTACHMENT,
                }
            )
        )

    def update_uniform_buffer(self):
        buffer = js.Uint8Array.new(bytes(self.uniforms))
        self.device.queue.writeBuffer(self.uniform_buffer, 0, buffer)

    def get_bindings(self):
        """Returns layout and resource arrays used to create binding layout and binding groups
        Current entires are: Uniforms, colormap texture, colormap sampler"""

        FRAGMENT = js.GPUShaderStage.FRAGMENT
        BOTH = js.GPUShaderStage.VERTEX | FRAGMENT

        layouts = [
            {
                "visibility": BOTH,
                "buffer": {"type": "uniform"},
            },
            {
                "visibility": FRAGMENT,
                "texture": {
                    "sampleType": "float",
                    "viewDimension": "1d",
                    "multisamples": False,
                },
            },
            {
                "visibility": FRAGMENT,
                "sampler": {"type": "filtering"},
            },
        ]
        resources = [
            {"resource": res}
            for res in [
                {"buffer": self.uniform_buffer},
                self.colormap_texture.createView(),
                self.colormap_sampler,
            ]
        ]
        return layouts, resources

    def begin_render_pass(self, command_encoder):
        render_pass_encoder = command_encoder.beginRenderPass(
            to_js(
                {
                    "colorAttachments": [
                        {
                            "view": self.context.getCurrentTexture().createView(),
                            "clearValue": {"r": 1, "g": 1, "b": 1, "a": 1},
                            "loadOp": "clear",
                            "storeOp": "store",
                        }
                    ],
                    "depthStencilAttachment": {
                        "view": self.depth_texture.createView(
                            to_js({"format": self.depth_format, "aspect": "all"})
                        ),
                        "depthLoadOp": "clear",
                        "depthStoreOp": "store",
                        "depthClearValue": 1.0,
                    },
                },
            )
        )
        render_pass_encoder.setViewport(
            0, 0, self.canvas.width, self.canvas.height, 0.0, 1.0
        )
        return render_pass_encoder


class MeshRenderObject:
    """Class that creates and manages all webgpu data structures to render a Netgen mesh"""

    def __init__(self, mesh, gpu, shader_code):
        self.mesh = mesh
        self.gpu = gpu

        self._create_buffers()

        self._create_bind_group()
        self._create_pipelines(shader_code)

    def _create_buffers(self):
        m = self.mesh

        self.n_vertices = len(m.Points())
        self.n_trigs = len(m.Elements2D())
        self.n_edges = 3 * self.n_trigs

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
            trigs.append(t.index)

        data = {
            "vertices": js.Float32Array.new(vertices),
            "edges": js.Int32Array.new(edges),
            "trigs": js.Int32Array.new(trigs),
        }

        buffers = {}
        for name, values in data.items():
            buffers[name] = self.gpu.device.createBuffer(
                to_js(
                    {
                        "size": values.length * 4,
                        "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
                    }
                )
            )
            self.gpu.device.queue.writeBuffer(buffers[name], 0, values)
        self._buffers = buffers

    def _create_bind_group(self):
        """Get binding data from WebGPU class and add values used for mesh rendering"""
        VERTEX = js.GPUShaderStage.VERTEX
        FRAGMENT = js.GPUShaderStage.FRAGMENT
        BOTH = VERTEX | FRAGMENT

        layouts, resources = self.gpu.get_bindings()

        for name in ["vertices", "edges", "trigs"]:
            layouts.append(
                {
                    "visibility": BOTH,
                    "buffer": {"type": "read-only-storage"},
                }
            )
            resources.append({"resource": {"buffer": self._buffers[name]}})

        for i in range(len(layouts)):
            layouts[i]["binding"] = i
            resources[i]["binding"] = i

        self._bind_group_layout = self.gpu.device.createBindGroupLayout(
            to_js({"entries": layouts})
        )

        self._bind_group = self.gpu.device.createBindGroup(
            to_js(
                {
                    "layout": self._bind_group_layout,
                    "entries": resources,
                }
            )
        )

    def _create_pipeline_layout(self):
        self._pipeline_layout = self.gpu.device.createPipelineLayout(
            to_js({"bindGroupLayouts": [self._bind_group_layout]})
        )

    def _create_pipelines(self, shader_code):
        self._create_pipeline_layout()
        shader_module = self.gpu.device.createShaderModule(to_js({"code": shader_code}))
        edges_pipeline = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "layout": self._pipeline_layout,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexEdge",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentEdge",
                        "targets": [{"format": self.gpu.format}],
                    },
                    "primitive": {"topology": "line-list"},
                    "depthStencil": {
                        **self.gpu.depth_stencil,
                    },
                }
            )
        )

        trigs_pipeline = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "layout": self._pipeline_layout,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexTrig",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentTrig",
                        "targets": [{"format": self.gpu.format}],
                    },
                    "primitive": {"topology": "triangle-list"},
                    "depthStencil": {
                        **self.gpu.depth_stencil,
                        # shift trigs behind to ensure that edges are rendered properly
                        "depthBias": 1.0,
                        "depthBiasSlopeScale": 1,
                    },
                }
            )
        )

        self.pipelines = {
            "edges": edges_pipeline,
            "trigs": trigs_pipeline,
        }

    def draw(self, encoder):
        encoder.setPipeline(self.pipelines["edges"])
        encoder.setBindGroup(0, self._bind_group)
        encoder.draw(2, self.n_edges, 0, 0)

        encoder.setPipeline(self.pipelines["trigs"])
        encoder.setBindGroup(0, self._bind_group)
        encoder.draw(3, self.n_trigs, 0, 0)


def to_js(value):
    return pyodide.ffi.to_js(value, dict_converter=js.Object.fromEntries)


def abort():
    js.alert("WebGPU is not supported")
    sys.exit(1)


def init_canvas(canvas):
    if not js.navigator.gpu:
        abort()

    if canvas is None:
        canvas = js.document.getElementById("canvas")

    # cloning and replacing the canvas removes all old event listeners
    new_canvas = canvas.cloneNode(True)
    canvas.parentNode.replaceChild(new_canvas, canvas)
    canvas = new_canvas
    del new_canvas

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


async def main(canvas=None, shader_url="./shader.wgsl"):
    canvas = init_canvas(canvas)
    adapter = await js.navigator.gpu.requestAdapter()

    if not adapter:
        abort()

    device = await adapter.requestDevice()

    gpu = WebGPU(device, canvas)
    input_handler = InputHandler(gpu)

    from netgen.occ import unit_square

    mesh = unit_square.GenerateMesh(maxh=0.2)
    shader_code = await (await js.fetch(shader_url)).text()
    mesh_object = MeshRenderObject(mesh, gpu, shader_code)

    # move mesh to center and scale it
    for i in [0, 5, 10]:
        gpu.uniforms.mat[i] = 1.8

    gpu.uniforms.mat[15] = 1.0

    # translate to center
    gpu.uniforms.mat[12] = -0.5 * 1.8
    gpu.uniforms.mat[13] = -0.5 * 1.8

    def render(time):
        gpu.update_uniform_buffer()

        command_encoder = device.createCommandEncoder()

        render_pass_encoder = gpu.begin_render_pass(command_encoder)
        mesh_object.draw(render_pass_encoder)
        render_pass_encoder.end()

        device.queue.submit([command_encoder.finish()])

    gpu.render_function = create_proxy(render)

    js.requestAnimationFrame(gpu.render_function)


main
