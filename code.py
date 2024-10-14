import js
from js import document, Float32Array
import pyodide.ffi

def to_js(value):
    return pyodide.ffi.to_js(value, dict_converter = js.Object.fromEntries)

async def main():
    if not js.navigator.gpu:
        js.alert(
            "WebGPU is not supported, see https://webgpu.io or use Chrome Canary with chrome://flags/#enable-unsafe-webgpu"
        )
        return
    
    adapter = await js.navigator.gpu.requestAdapter()
    device = await adapter.requestDevice()
    format = js.navigator.gpu.getPreferredCanvasFormat()
    
    # Create the render pipeline
    render_pipeline = device.createRenderPipeline(to_js({
        "layout": "auto",
        "vertex": {
            "module": device.createShaderModule(to_js({"code": vertexShader})),
            "entryPoint": "VSMain",
        },
        "fragment": {
            "module": device.createShaderModule(to_js({"code": fragmentShader})),
            "entryPoint": "PSMain",
            "targets": [{"format": format}],
        },
        "primitive": {"topology": "triangle-list"},
    }))
    
    # Get the canvas context and configure it
    context = document.getElementById("canvas").getContext("webgpu")
    context.configure(to_js({
        "device": device,
        "format": format,
        "alphaMode": "premultiplied",
    }))
    
    # Create the uniform buffer
    uniform_buffer = device.createBuffer(to_js({
        "size": 4,
        "usage": js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST,
    }))
    
    # Create the bind group for the uniforms
    uniform_bind_group = device.createBindGroup(to_js({
        "layout": render_pipeline.getBindGroupLayout(0),
        "entries": [{"binding": 0, "resource": {"buffer": uniform_buffer}}],
    }))
    
    uniforms = Float32Array.new(1)
    
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
        
        render_pass_encoder = command_encoder.beginRenderPass(to_js({
            "colorAttachments": [render_pass_color_attachment],
        }))
        
        render_pass_encoder.setPipeline(render_pipeline)
        render_pass_encoder.setBindGroup(0, uniform_bind_group)
        render_pass_encoder.draw(3, 1, 0, 0)
        render_pass_encoder.end()
        
        device.queue.submit([command_encoder.finish()])
        js.requestAnimationFrame(pyodide.ffi.create_proxy(update))
    
    js.requestAnimationFrame(pyodide.ffi.create_proxy(update))

vertexShader = """
				struct Interpolators 
				{
					@builtin(position) position: vec4<f32>,
					@location(0) texcoord: vec2<f32>,
				};

				@vertex
				fn VSMain(@builtin(vertex_index) vertexId: u32) -> Interpolators
				{
					let vertices = array<vec2<f32>,3> (vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
					return Interpolators(vec4(vertices[vertexId], 0.0, 1.0), vec2(0.5 * vertices[vertexId] + vec2(0.5, 0.5)));
				}
"""
fragmentShader = """
				struct Uniforms {iTime : f32};
				@group(0) @binding(0) var<uniform> uniforms : Uniforms;

				fn Triangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32
				{
					var ba = b - a;
					var cb = c - b;
					var ac = a - c;
					var pa = p - a;
					var pb = p - b;
					var pc = p - c;
					var q0: vec2<f32> = pa - ba * clamp( dot(pa,ba) / dot(ba,ba), 0.0, 1.0);
					var q1: vec2<f32> = pb - cb * clamp( dot(pb,cb) / dot(cb,cb), 0.0, 1.0);
					var q2: vec2<f32> = pc - ac * clamp( dot(pc,ac) / dot(ac,ac), 0.0, 1.0);   
					var s: f32 = ba.x * ac.y - ba.y * ac.x;
					var d = vec2<f32>(dot(q0, q0), s * (pa.x * ba.y - pa.y * ba.x));
					d = min(d, vec2<f32>(dot(q1, q1), s * (pb.x * cb.y - pb.y * cb.x)));
					d = min(d, vec2<f32>(dot(q2, q2), s * (pc.x * ac.y - pc.y * ac.x)));
					return step(-sqrt(d.x) * sign(d.y), 0.0);
				}

				@fragment
				fn PSMain(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> 
				{
					let t1 = Triangle(texcoord, vec2(0.050, 0.800), vec2(0.675, 0.800), vec2(0.350, 0.250));
					let t2 = Triangle(texcoord, vec2(0.675, 0.800), vec2(0.825, 0.525), vec2(0.512, 0.524));
					let t3 = Triangle(texcoord, vec2(0.513, 0.524), vec2(0.825, 0.525), vec2(0.675, 0.250));
					let t4 = Triangle(texcoord, vec2(0.746, 0.670), vec2(0.825, 0.800), vec2(0.910, 0.670));
					let t5 = Triangle(texcoord, vec2(0.746, 0.670), vec2(0.910, 0.670), vec2(0.825, 0.525));
					let c1 = vec3(0.00, 0.35, 0.61) * t1;
					let c2 = vec3(0.00, 0.40, 0.70) * t2;
					let c3 = vec3(0.00, 0.46, 0.80) * t3;
					let c4 = vec3(0.00, 0.57, 1.00) * t4;
					let c5 = vec3(0.00, 0.52, 0.91) * t5;
					let logo = c1 + c2 + c3 + c4 + c5;
					let f = logo.r > 0.0 || logo.g > 0.0 || logo.b > 0.0;
					return mix(vec4(logo, 1.0), vec4(vec3(sin(uniforms.iTime) * 0.5 + 0.5), 1.0), step(f32(f), 0.0));
				}
"""

# Run the main function
await main()

