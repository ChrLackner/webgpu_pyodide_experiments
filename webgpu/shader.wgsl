struct Edge { v: vec2<u32> };

struct Segment { v: vec2<u32>, index: i32 };
struct Trig { v: vec3<u32>, index: i32 };

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
@group(0) @binding(6) var<storage> trig_function_values : array<f32>;

struct VertexOutput1d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: f32,
  @location(2) id: u32,
};

struct VertexOutput2d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec2<f32>,
  @location(2) id: u32,
};

struct VertexOutput3d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec3<f32>,
  @location(2) id: u32,
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
fn mainVertexTrig(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    var trig = trigs[trigId];
    var vid: u32 = trig.v[vertexId];
    var p = vertices[vid];
    var position = calcPosition(p);
    var lam: vec2<f32> = vec2<f32>(0.);
    if vertexId < 2 {
        lam[vertexId] = 1.0;
    }
    return VertexOutput2d(position, p, lam, trigId);
}

@vertex
fn mainVertexEdge(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) edgeId: u32) -> VertexOutput1d {
    var edge = edges[edgeId];
    var vid: u32 = edge.v[vertexId];
    var p = vertices[vid];
    var position = calcPosition(p);
    var lam: f32 = 0.;
    if vertexId == 0 {
        lam = 1.0;
    }
    return VertexOutput1d(position, p, lam, edgeId);
}

@fragment
fn mainFragmentTrig(@location(0) p: vec3<f32>, @location(1) lam: vec2<f32>, @location(2) id: u32) -> @location(0) vec4<f32> {
    let verts = trigs[id].v;
    let v0 = trig_function_values[ 3 * id ];
    let v1 = trig_function_values[ 3 * id + 1];
    let v2 = trig_function_values[ 3 * id + 2];

    checkClipping(p);

    let value = evalTrigP1(array(v0, v1, v2), lam);
    return getColor(value);
}

@fragment
fn mainFragmentEdge(@location(0) p: vec3<f32>) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}

fn evalSegP1(values: array<f32, 2>, lam: f32) -> f32 {
    return mix(values[0], values[1], lam);
}

fn evalTrigP1(values: array<f32, 3>, lam: vec2<f32>) -> f32 {
    return values[0] * lam.x + values[1] * lam.y + values[2] * (1.0 - lam.x - lam.y);
}

fn evalTetP1(values: array<f32, 4>, lam: vec3<f32>) -> f32 {
    return values[0] * lam.x + values[1] * lam.y + values[2] * lam.z + values[3] * (1.0 - lam.x - lam.y - lam.z);
}

////////////////////////////////////////////////////////////////////////////////////////
// Generated code

fn evalSegP2Basis(x: f32) -> array<f32, 3> {
    let y = 1.0 - x;
    return array(x * x, 2.0 * x * y, y * y);
}

fn evalSegP2(v: array<f32, 3>, lam: f32) -> f32 {
    let basis = evalSegP2Basis(lam);
    var result: f32 = basis[0] * v[0];
    for (var i: u32 = 1; i < 3; i++) {
        result += basis[i] * v[i];
    }
    return result;
}

fn evalTrigP2Basis(lam: vec2<f32>) -> array<f32, 6> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x - y;
    let x0 = 2.0 * x;
    return array(x * x, x0 * y, y * y, x0 * z, 2.0 * y * z, z * z);
}

fn evalTrigP2(v: array<f32, 6>, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP2Basis(lam);
    var result: f32 = basis[0] * v[0];
    for (var i: u32 = 1; i < 6; i++) {
        result += basis[i] * v[i];
    }
    return result;
}

