import math

import js
import ngsolve as ngs
import ngsolve.webgui
import numpy as np

from .uniforms import Binding
from .utils import to_js


class MeshRenderObject:
    """Class that creates and manages all webgpu data structures to render an NGSolve mesh with a coefficient function"""

    def __init__(self, gpu):
        self.gpu = gpu

    def draw1(self, cf, region, order=1):
        """Draw the coefficient function on a region"""
        self.n_trigs = len(region.mesh.ngmesh.Elements2D())
        device = self.gpu.device

        buffers = create_mesh_buffers(device, region, curve_order=1)
        buffers.update(create_function_value_buffers(device, cf, region, order))
        self._buffers = buffers

        self._create_bind_group()
        self._create_pipelines()

    def draw2(self, N):
        x_range = np.linspace(0, 1, N, dtype=np.float32)
        y_range = np.linspace(0, 1, N, dtype=np.float32)

        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.zeros_like(xx)
        points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # top_left = np.arange((N - 1) * (N - 1)).reshape((N - 1, N - 1))
        # bottom_left = top_left + N
        # top_right = top_left + 1
        # bottom_right = bottom_left + 1
        indices = np.arange(N * N).reshape(N, N)

        top_left = indices[:-1, :-1].ravel()
        top_right = indices[:-1, 1:].ravel()
        bottom_left = indices[1:, :-1].ravel()
        bottom_right = indices[1:, 1:].ravel()

        tri1 = np.vstack(
            [top_left.ravel(), bottom_left.ravel(), bottom_right.ravel()]
        ).T
        tri2 = np.vstack([top_left.ravel(), bottom_right.ravel(), top_right.ravel()]).T
        all_triangles = np.vstack([tri1, tri2])
        # all_triangles = tri1

        n_trigs = all_triangles.shape[0]
        self.n_trigs = n_trigs

        p_trigs = points[all_triangles]

        trigs = np.zeros(
            n_trigs,
            dtype=[
                ("p", np.float32, 9),  # 3 vec3<f32> (each 4 floats due to padding)
                ("index", np.int32),  # index (i32)
            ],
        )
        trigs["p"] = p_trigs.reshape(-1, 9)
        trigs["index"] = [1] * n_trigs
        data = js.Uint8Array.new(trigs.tobytes())

        trigs_buffer = self.gpu.device.createBuffer(
            to_js(
                {
                    "size": data.length,
                    "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
                }
            )
        )
        self.gpu.device.queue.writeBuffer(trigs_buffer, 0, data)

        function_buffer = self.gpu.device.createBuffer(
            to_js(
                {
                    "size": 4 * (3 * self.n_trigs + 2),
                    "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
                }
            )
        )
        data = js.Uint8Array.new(
            np.concatenate(
                (
                    np.array([1, 1], dtype=np.float32),
                    np.linspace(0, 1, 3 * self.n_trigs, dtype=np.float32),
                )
            ).tobytes()
        )
        self.gpu.device.queue.writeBuffer(function_buffer, 0, data)
        self._buffers = {"trigs": trigs_buffer, "trig_function_values": function_buffer}
        print("n_trigs", self.n_trigs)

        self._create_bind_group()
        self._create_pipelines()

    def create_testing_square_mesh(self, n):
        # launch compute shader
        n = math.ceil(n / 32) * 32
        n_trigs = n * n
        self.n_trigs = n_trigs
        print(f"n_trigs {n_trigs/10**6:.3f}, M")
        trig_size = 4 * n_trigs * 9
        value_size = 4 * (3 * n_trigs + 2)
        print(f"trig size {trig_size/1024/1024:.2f} MB")
        print(f"vals size {value_size/1024/1024:.2f} MB")
        trigs_buffer = self.gpu.device.createBuffer(
            to_js(
                {
                    "size": 4 * n_trigs * 10,
                    "usage": js.GPUBufferUsage.STORAGE,
                }
            )
        )
        function_buffer = self.gpu.device.createBuffer(
            to_js(
                {
                    "size": 4 * (3 * n_trigs + 2),
                    "usage": js.GPUBufferUsage.STORAGE,
                }
            )
        )
        self._bind_group_layout = self.gpu.device.createBindGroupLayout(
            to_js(
                {
                    "entries": [
                        {
                            "binding": 5,
                            "visibility": js.GPUShaderStage.COMPUTE,
                            "buffer": {"type": "storage"},
                        },
                        {
                            "binding": 6,
                            "visibility": js.GPUShaderStage.COMPUTE,
                            "buffer": {"type": "storage"},
                        },
                    ]
                }
            )
        )
        self._bind_group = self.gpu.device.createBindGroup(
            to_js(
                {
                    "layout": self._bind_group_layout,
                    "entries": [
                        {
                            "binding": 5,
                            "resource": {"buffer": trigs_buffer},
                        },
                        {
                            "binding": 6,
                            "resource": {"buffer": function_buffer},
                        },
                    ],
                }
            )
        )

        device = self.gpu.device
        shader_code = open("webgpu/compute.wgsl").read()
        shader_module = device.createShaderModule(to_js({"code": shader_code}))
        pipeline = device.createComputePipeline(
            to_js(
                {
                    "layout": device.createPipelineLayout(
                        to_js({"bindGroupLayouts": [self._bind_group_layout]})
                    ),
                    "compute": {"module": shader_module, "entryPoint": "create_mesh"},
                }
            )
        )

        command_encoder = device.createCommandEncoder()
        pass_encoder = command_encoder.beginComputePass()
        pass_encoder.setPipeline(pipeline)
        pass_encoder.setBindGroup(0, self._bind_group)

        pass_encoder.dispatchWorkgroups(n // 32, 1, 1)
        pass_encoder.end()
        device.queue.submit([command_encoder.finish()])

        self._buffers = {
            "trigs": trigs_buffer,
            "trig_function_values": function_buffer,
        }

        self._create_bind_group()
        self._create_pipelines()

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
        #                 "entryPoint": "mainVertexEdgeP1",
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

    def render(self, encoder):
        # encoder.setPipeline(self.pipelines["edges"])
        # encoder.setBindGroup(0, self._bind_group)
        # encoder.draw(2, 3 * self.n_trigs, 0, 0)
        #
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


def create_mesh_buffers(device, region, curve_order=1):
    """Create buffers for the mesh geometry"""
    # TODO: implement other element types than triangles
    # TODO: handle region correctly to draw only part of the mesh
    # TODO: handle 3d meshes correctly
    # TODO: set up proper index buffer
    mesh = region.mesh
    points = evaluate_cf(ngs.CF((ngs.x, ngs.y, ngs.z)), mesh.Region(ngs.VOL), order=1)

    n_trigs = len(mesh.ngmesh.Elements2D())

    edge_points = points[2:].reshape(-1, 3, 3)
    edges = np.zeros((n_trigs, 3, 2, 3), dtype=np.float32)
    for i in range(3):
        edges[:, i, 0, :] = edge_points[:, i, :]
        edges[:, i, 1, :] = edge_points[:, (i + 1) % 3, :]
    edge_data = js.Uint8Array.new(edges.flatten().tobytes())
    edge_buffer = device.createBuffer(
        to_js(
            {
                "size": edge_data.length,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )
    device.queue.writeBuffer(edge_buffer, 0, edge_data)

    trigs = np.zeros(
        n_trigs,
        dtype=[
            ("p", np.float32, 9),  # 3 vec3<f32> (each 4 floats due to padding)
            ("index", np.int32),  # index (i32)
        ],
    )
    trigs["p"] = points[2:].flatten().reshape(-1, 9)
    trigs["index"] = [1] * n_trigs
    data = js.Uint8Array.new(trigs.tobytes())

    trigs_buffer = device.createBuffer(
        to_js(
            {
                "size": data.length,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )
    device.queue.writeBuffer(trigs_buffer, 0, data)
    return {"trigs": trigs_buffer, "edges": edge_buffer}


def create_function_value_buffers(device, cf, region, order):
    """Evaluate a coefficient function on a mesh and create GPU buffer with the values,
    returns a dictionary with the buffer as value and the name/element type as key"""
    # TODO: implement other element types than triangles
    values = evaluate_cf(cf, region, order)
    data = js.Uint8Array.new(values.tobytes())
    buffer = device.createBuffer(
        to_js(
            {
                "size": data.length,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )
    device.queue.writeBuffer(buffer, 0, data)
    return {"trig_function_values": buffer}


def evaluate_cf(cf, region, order):
    """Evaluate a coefficient function on a mesh and returns the values as a flat array, ready to copy to the GPU as storage buffer.
    The first two entries are the function dimension and the polynomial order of the stored values.
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

    pts = region.mesh.MapToAllElements(
        {ngs.ET.TRIG: intrule, ngs.ET.QUAD: intrule}, region
    )
    pmat = cf(pts)
    pmat = pmat.reshape(-1, ndof, comps)

    values = np.zeros((ndof, pmat.shape[0], comps), dtype=np.float32)
    for i in range(comps):
        ngsmat = ngs.Matrix(pmat[:, :, i].transpose())
        values[:, :, i] = ibmat * ngsmat

    values = values.transpose((1, 0, 2)).flatten()
    ret = np.concatenate(([np.float32(cf.dim), np.float32(order)], values))
    return ret
