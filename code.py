import js
from js import document, Float32Array
import pyodide.ffi
import sys


def to_js(value):
    return pyodide.ffi.to_js(value, dict_converter=js.Object.fromEntries)


def abort():
    js.alert("WebGPU is not supported")
    sys.exit(1)


async def main():
    if not js.navigator.gpu:
        abort()

    adapter = await js.navigator.gpu.requestAdapter()
    if not adapter:
        abort()

    device = await adapter.requestDevice()
    format = js.navigator.gpu.getPreferredCanvasFormat()

    # Get the canvas context and configure it
    context = document.getElementById("canvas").getContext("webgpu")
    context.configure(
        to_js(
            {
                "device": device,
                "format": format,
                "alphaMode": "premultiplied",
            }
        )
    )

    # Create the uniform buffer
    uniform_buffer = device.createBuffer(
        to_js(
            {
                "size": 4,
                "usage": js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST,
            }
        )
    )

    # Create the storage buffer
    storage_buffer = device.createBuffer(
        to_js(
            {
                "size": 4 * 4 * 3,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )

    bindGroupLayout = device.createBindGroupLayout(
        to_js(
            {
                "entries": [
                    {
                        "binding": 0,
                        "visibility": js.GPUShaderStage.FRAGMENT,
                        "buffer": {"type": "uniform"},
                    },
                    {
                        "binding": 1,
                        "visibility": js.GPUShaderStage.VERTEX,
                        "buffer": {"type": "read-only-storage"},
                    },
                ],
            }
        )
    )

    # Create the bind group for the uniforms
    uniform_bind_group = device.createBindGroup(
        to_js(
            {
                "layout": bindGroupLayout,
                "entries": [
                    {"binding": 0, "resource": {"buffer": uniform_buffer}},
                    {"binding": 1, "resource": {"buffer": storage_buffer}},
                ],
            }
        )
    )

    pipelineLayout = device.createPipelineLayout(
        to_js({"bindGroupLayouts": [bindGroupLayout]})
    )

    # Create the render pipeline
    render_pipeline = device.createRenderPipeline(
        to_js(
            {
                "layout": pipelineLayout,
                "vertex": {
                    "module": device.createShaderModule(to_js({"code": vertexShader})),
                    "entryPoint": "VSMain",
                },
                "fragment": {
                    "module": device.createShaderModule(
                        to_js({"code": fragmentShader})
                    ),
                    "entryPoint": "PSMain",
                    "targets": [{"format": format}],
                },
                "primitive": {"topology": "triangle-list"},
            }
        )
    )

    uniforms = Float32Array.new(1)
    vertices = Float32Array.new(12)
    for i, val in enumerate(
        [
            0,
            0,
            0,
            0.03,
            1,
            0,
            0,
            0.2,
            0,
            1,
            0,
            1.0,
        ]
    ):
        vertices[i] = float(val)
    device.queue.writeBuffer(storage_buffer, 0, vertices)

    def update(time):
        uniforms[0] = time * 0.001
        device.queue.writeBuffer(uniform_buffer, 0, uniforms)

        command_encoder = device.createCommandEncoder()
        texture_view = context.getCurrentTexture().createView()

        render_pass_color_attachment = {
            "view": texture_view,
            "clearValue": {"r": 0, "g": 0, "b": 0, "a": 1},
            "loadOp": "clear",
            "storeOp": "store",
        }

        render_pass_encoder = command_encoder.beginRenderPass(
            to_js(
                {
                    "colorAttachments": [render_pass_color_attachment],
                }
            )
        )

        render_pass_encoder.setPipeline(render_pipeline)
        render_pass_encoder.setBindGroup(0, uniform_bind_group)
        render_pass_encoder.draw(3, 1, 0, 0)
        render_pass_encoder.end()

        device.queue.submit([command_encoder.finish()])
        js.requestAnimationFrame(pyodide.ffi.create_proxy(update))

    js.requestAnimationFrame(pyodide.ffi.create_proxy(update))


vertexShader = """
				struct Vertex {
                        position : vec3<f32>,
                        value: f32,
                        };
				@group(0) @binding(1) var<storage> vertices : array<Vertex>;
				struct Interpolators 
				{
					@builtin(position) position: vec4<f32>,
					@location(0) color: vec3<f32>,
				};

				@vertex
				fn VSMain(@builtin(vertex_index) vertexId: u32) -> Interpolators
				{
					var position = vertices[vertexId].position;
					return Interpolators(vec4(position,  1.0), vec3<f32>(vertices[vertexId].value));
				}
"""
fragmentShader = """
				struct Uniforms {iTime : f32};
				@group(0) @binding(0) var<uniform> uniforms : Uniforms;

				@fragment
				fn PSMain(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> 
				{
                    return vec4<f32>(color, 1.0);
				}
"""

# Run the main function
await main()
