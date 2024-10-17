import js

from .uniforms import Binding
from .utils import to_js


class MeshRenderObject:
    """Class that creates and manages all webgpu data structures to render a Netgen mesh"""

    def __init__(self, mesh, gpu):
        self.mesh = mesh
        self.gpu = gpu

        self._create_buffers()

        self._create_bind_group()
        self._create_pipelines()

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
        trig_function_values = []
        for t in m.Elements2D():
            for i in range(3):
                trigs.append(t.vertices[i].nr - 1)
                edges.append(t.vertices[i].nr - 1)
                edges.append(t.vertices[(i + 1) % 3].nr - 1)
                trig_function_values.append(vertices[4 * (t.vertices[i].nr - 1)])
            trigs.append(t.index)

        data = {
            "vertices": js.Float32Array.new(vertices),
            "edges": js.Int32Array.new(edges),
            "trigs": js.Int32Array.new(trigs),
            "trig_function_values": js.Float32Array.new(trig_function_values),
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

    def get_binding_layout(self):
        layouts = []
        for name in self._buffers.keys():
            binding = getattr(Binding, name.upper())
            layouts.append(
                {
                    "binding": binding,
                    "visibility": js.GPUShaderStage.FRAGMENT | js.GPUShaderStage.VERTEX,
                    "buffer": {"type": "read-only-storage"},
                }
            )
        return layouts

    def get_binding(self):
        resources = []
        for name in self._buffers.keys():
            binding = getattr(Binding, name.upper())
            resources.append(
                {"binding": binding, "resource": {"buffer": self._buffers[name]}}
            )
        return resources

    def _create_bind_group(self):
        """Get binding data from WebGPU class and add values used for mesh rendering"""
        layouts = []
        resources = []

        # gather binding layouts and resources from all objects
        for obj in [self.gpu.uniforms, self.gpu.colormap, self]:
            layouts += obj.get_binding_layout()
            resources += obj.get_binding()

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

    def _create_pipelines(self):
        shader_code = open("webgpu/shader.wgsl").read()
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

    def __del__(self):
        for buffer in self._buffers.values():
            buffer.destroy()
