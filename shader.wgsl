
struct Normal { n : vec3<f32> };
struct Trig { v : vec3<u32>, index: i32 };
struct Edge { v : vec2<u32> };


@group(0) @binding(1) var<storage> vertices : array<vec3<f32>>;
@group(0) @binding(2) var<storage> trigs : array<Trig>;
@group(0) @binding(3) var<storage> edges : array<Edge>;

struct VertexOutput
{
  @builtin(position) position: vec4<f32>,
  @location(0) lam: vec3<f32>,
};

fn calcPosition(p: vec3<f32>) -> vec3<f32> {
  var out = p - vec3<f32>(0.5, 0.5, 0.0);
  return 2*out;
}

@vertex
fn mainVertexTrig(@builtin(vertex_index) vertexId: u32) -> VertexOutput
{
  var trigId = vertexId / 3;
  var trig = trigs[trigId];
  var vid = trig.v[vertexId % 3];
  var position = calcPosition(vertices[vid]);
  var lam: vec3<f32> = vec3<f32>(0.);
  if(vertexId % 3 == 0) { lam[0] = 1.0; }
  else if(vertexId % 3 == 1) { lam[1] = 1.0; }
  else  { lam[2] = 1.0; }
  return VertexOutput(vec4(position,  1.0), lam);
}

@vertex
fn mainVertexEdge(@builtin(vertex_index) vertexId: u32) -> VertexOutput
{
  var edgeId = vertexId / 2;
  var edge = edges[edgeId];
  var vid = edge.v[vertexId % 2];
  var position = calcPosition(vertices[vid]);
  var lam: vec3<f32> = vec3<f32>(0.);
  if(vertexId % 2 == 0) { lam[0] = 1.0; }
  else { lam[1] = 1.0;}
  return VertexOutput(vec4(position,  1.0), lam);
}

struct Uniforms {iTime : f32};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@fragment
fn mainFragment(@location(0) lam: vec3<f32>) -> @location(0) vec4<f32> 
{
  var value = 0.0;
  if(min(min(lam[0], lam[1]), 1.0-lam[0]-lam[1]) < 0.01) {
    value = 1.0;
  }
  return vec4<f32>(value, value, value, 1.0);
}
