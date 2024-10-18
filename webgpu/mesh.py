import math

import js
import ngsolve as ngs
import ngsolve.webgui
import numpy as np

from .uniforms import Binding
from .utils import to_js


class MeshRenderObject:
    """Class that creates and manages all webgpu data structures to render a Netgen mesh"""

    def __init__(self, gpu):
        self.gpu = gpu

    def fill_buffers(self, gfu, order=1):
        self._create_buffers(gfu, order=order)
        self._create_bind_group()
        self._create_pipelines()

    def _create_buffers(self, gfu, order):
        m = gfu.space.mesh.ngmesh
        self.n_trigs = len(m.Elements2D())

        function_values = evaluate_cf(gfu, gfu.space.mesh, order=order)
        vertex_coordinates = evaluate_cf(
            ngs.CF((ngs.x, ngs.y, ngs.z)), gfu.space.mesh, order=1
        )
        trig_points = vertex_coordinates

        trigs = np.zeros(
            self.n_trigs,
            dtype=[
                ("p", np.float32, 9),  # 3 vec3<f32> (each 4 floats due to padding)
                ("index", np.int32),  # index (i32)
            ],
        )
        trigs["p"] = trig_points[2:].flatten().reshape(-1, 9)
        trigs["index"] = [1] * self.n_trigs
        trigs = trigs.tobytes()

        to_u8 = lambda x: js.Uint8Array.new(x.buffer)

        data = {
            "trigs": js.Uint8Array.new(trigs),
            "trig_function_values": to_u8(js.Float32Array.new(function_values)),
        }

        buffers = {}
        for name, values in data.items():
            buffers[name] = self.gpu.device.createBuffer(
                to_js(
                    {
                        "size": values.length,
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
        shader_code = (
            open("webgpu/shader.wgsl").read() + open("webgpu/eval.wgsl").read()
        )
        self._create_pipeline_layout()
        shader_module = self.gpu.device.createShaderModule(to_js({"code": shader_code}))
        # edges_pipeline = self.gpu.device.createRenderPipeline(
        #     to_js(
        #         {
        #             "layout": self._pipeline_layout,
        #             "vertex": {
        #                 "module": shader_module,
        #                 "entryPoint": "mainVertexEdge",
        #             },
        #             "fragment": {
        #                 "module": shader_module,
        #                 "entryPoint": "mainFragmentEdge",
        #                 "targets": [{"format": self.gpu.format}],
        #             },
        #             "primitive": {"topology": "line-list"},
        #             "depthStencil": {
        #                 **self.gpu.depth_stencil,
        #             },
        #         }
        #     )
        # )

        trigs_pipeline = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "layout": self._pipeline_layout,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexTrigP1",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentTrig",
                        "targets": [{"format": self.gpu.format}],
                    },
                    "primitive": {
                        "topology": "triangle-list",
                        "cullMode": "none",
                        "frontFace": "ccw",
                    },
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
            # "edges": edges_pipeline,
            "trigs": trigs_pipeline,
        }

    def draw(self, encoder):
        # encoder.setPipeline(self.pipelines["edges"])
        # encoder.setBindGroup(0, self._bind_group)
        # encoder.draw(2, self.n_edges, 0, 0)

        encoder.setPipeline(self.pipelines["trigs"])
        encoder.setBindGroup(0, self._bind_group)
        encoder.draw(3, self.n_trigs, 0, 0)

    def __del__(self):
        for buffer in self._buffers.values():
            buffer.destroy()


def _get_bernstein_matrix_trig(n, intrule):
    """Create inverse vandermonde matrix for the Bernstein basis functions on a triangle of degree n and given integration points"""
    ndtrig = int((n + 1) * (n + 2) / 2)

    mat = ngs.Matrix(ndtrig, ndtrig)
    fac_n = math.factorial(n)
    for row, ip in enumerate(intrule):
        col = 0
        x = ip.point[0]
        y = ip.point[1]
        z = 1.0 - x - y
        for i in range(n + 1):
            factor = fac_n / math.factorial(i) * x**i
            for j in range(n + 1 - i):
                k = n - i - j
                factor2 = 1.0 / (math.factorial(j) * math.factorial(k))
                mat[row, col] = factor * factor2 * y**j * z**k
                col += 1
    return mat


def evaluate_cf(cf, mesh, order):
    """Evaluate a coefficient function on a mesh and return the values as a flat array, ready to copy to the GPU
    The first two values are the dimension and the order of the coefficient function, followed by the values
    """
    comps = cf.dim
    int_points = ngsolve.webgui._make_trig(order)
    intrule = ngs.IntegrationRule(
        int_points,
        [
            0,
        ]
        * len(int_points),
    )
    ibmat = _get_bernstein_matrix_trig(order, intrule).I

    ndof = ibmat.h

    pts = mesh.MapToAllElements({ngs.ET.TRIG: intrule, ngs.ET.QUAD: intrule}, ngs.VOL)
    pmat = cf(pts)
    pmat = pmat.reshape(-1, ndof, comps)

    values = np.zeros((ndof, pmat.shape[0], comps))
    for i in range(comps):
        ngsmat = ngs.Matrix(pmat[:, :, i].transpose())
        values[:, :, i] = ibmat * ngsmat

    values = values.transpose((1, 0, 2)).flatten()
    return np.concatenate(([float(cf.dim), float(order)], values))
