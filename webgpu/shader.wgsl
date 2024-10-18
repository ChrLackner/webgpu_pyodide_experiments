struct Edge { v: vec2<u32> };

struct Segment { v: vec2<u32>, index: i32 };
struct TrigP1 { p: array<f32, 9>, index: i32 }; // 3 vertices with 3 coordinates each, don't use vec3 due to 16 byte alignment
struct TrigP2 { p: array<f32, 18>, index: i32 };

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

const VALUES_OFFSET: u32 = 2; // storing number of components and order of basis functions in first two entries

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var colormap : texture_1d<f32>;
@group(0) @binding(2) var colormap_sampler : sampler;

// @group(0) @binding(4) var<storage> edges : array<Edge>;
@group(0) @binding(5) var<storage> trigs_p1 : array<TrigP1>;
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
fn mainVertexTrigP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    let v_offset = 3 * (3 * trigId + vertexId) + VALUES_OFFSET;

    let trig = trigs_p1[trigId];
    var p = vec3<f32>(trig.p[3 * vertexId], trig.p[3 * vertexId + 1], trig.p[3 * vertexId + 2]);

    var lam: vec2<f32> = vec2<f32>(0.);
    if vertexId < 2 {
        lam[vertexId] = 1.0;
    }

    var position = calcPosition(p);

    return VertexOutput2d(position, p, lam, trigId);
}

@fragment
fn mainFragmentTrig(@location(0) p: vec3<f32>, @location(1) lam: vec2<f32>, @location(2) id: u32) -> @location(0) vec4<f32> {
    checkClipping(p);
    let value = evalTrig(id, 0u, lam);
    return getColor(value);
}

@fragment
fn mainFragmentEdge(@location(0) p: vec3<f32>) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}

