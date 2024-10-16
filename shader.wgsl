struct Edge { v: vec2<u32> };

struct Segment { v: vec2<u32>, index: i32 };
struct Trig { v: vec3<u32>, index: i32 };
struct Quad { v: vec4<u32>, index: i32 };

struct ClippingPlane {
  normal: vec3<f32>,
  dist: f32,
};

struct Colormap {
  min: f32,
  max: f32,
}

struct Complex {
  re: f32,
  imag: f32,
};

struct Uniforms {
  mat: mat4x4<f32>,
  clipping_plane: vec4<f32>,
  colormap: vec2<f32>,
  scaling: vec2<f32>,
  aspect: f32,
  eval_mode: u32,
  do_clipping: u32,
  padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var colormap : texture_1d<f32>;
@group(0) @binding(2) var colormap_sampler : sampler;

@group(0) @binding(3) var<storage> vertices : array<vec3<f32>>;
@group(0) @binding(4) var<storage> edges : array<Edge>;
@group(0) @binding(5) var<storage> trigs : array<Trig>;

struct VertexOutput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) value: f32,
  @location(2) lam: vec3<f32>,
};

fn calcPosition(p: vec3<f32>) -> vec4<f32> {
    return uniforms.mat * vec4<f32>(p, 1.0);
}

fn checkClipping(p: vec3<f32>) {
    if uniforms.do_clipping != 0 {
        if dot(uniforms.clipping_plane, vec4<f32>(p, 1.0)) < 0 {
        discard;
        }
    }
}

fn getColor(value: f32) -> vec4<f32> {
    return textureSample(colormap, colormap_sampler, value);
}

@vertex
fn mainVertexTrig(@builtin(vertex_index) vertexId: u32) -> VertexOutput {
    var trigId = vertexId / 3;
    var trig = trigs[trigId];
    var vid = trig.v[vertexId % 3];
    var p = vertices[vid];
    var position = calcPosition(p);
    position.z = 0.01;
    var lam: vec3<f32> = vec3<f32>(0.);
    if vertexId % 3 == 0 { lam[0] = 1.0; } else if vertexId % 3 == 1 { lam[1] = 1.0; } else { lam[2] = 1.0; }
    var value: f32 = p.x;
    return VertexOutput(position, p, value, lam);
}

@vertex
fn mainVertexEdge(@builtin(vertex_index) vertexId: u32) -> VertexOutput {
    var edgeId = vertexId / 2;
    var edge = edges[edgeId];
    var vid = edge.v[vertexId % 2];
    var p = vertices[vid];
    var position = calcPosition(p);
    var lam: vec3<f32> = vec3<f32>(0.);
    if vertexId % 2 == 0 { lam[0] = 1.0; } else { lam[1] = 1.0;}
    var value: f32 = p.x; // todo: evaluate function here
    return VertexOutput(position, p, value, lam);
}

@fragment
fn mainFragmentTrig(@location(0) p: vec3<f32>, @location(1) value: f32) -> @location(0) vec4<f32> {
    checkClipping(p);
    return getColor(value);
}

@fragment
fn mainFragmentEdge(@location(0) p: vec3<f32>, @location(1) value: f32) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}
