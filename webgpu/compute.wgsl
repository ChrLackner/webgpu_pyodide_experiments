struct TrigP1 { p: array<f32, 9>, index: i32 };
@group(0) @binding(5) var<storage, read_write> trigs_p1 : array<TrigP1>;
@group(0) @binding(6) var<storage, read_write> trig_function_values : array<f32>;

@compute  @workgroup_size(16, 16, 1)
fn create_mesh(@builtin(num_workgroups) n_groups: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n: u32 = n_groups.x * 16;
    let h: f32 = 1.0 / (f32(n) + 1.);

    if lid.x == 0 && lid.y == 0 && wid.x == 0 {
        trig_function_values[0] = 1.0;
        trig_function_values[1] = 1.0;
    }

    for (var ix: u32 = wid.x * 16u + lid.x; ix < n; ix += 16u * n_groups.x) {
        let x: f32 = h * f32(ix);
        for (var iy: u32 = wid.y * 16u + lid.y; iy < n; iy += 16u) {
            let y: f32 = h * f32(iy);
            let i = ix + iy * n;
            trigs_p1[i].p[0] = x;
            trigs_p1[i].p[1] = y;
            trigs_p1[i].p[2] = 0.0;

            trigs_p1[i].p[3] = x + h;
            trigs_p1[i].p[4] = y;
            trigs_p1[i].p[5] = 0.0;

            trigs_p1[i].p[6] = x;
            trigs_p1[i].p[7] = y + h;
            trigs_p1[i].p[8] = 0.0;


            trig_function_values[2 + 3 * i + 0] = x;
            trig_function_values[2 + 3 * i + 1] = x + h;
            trig_function_values[2 + 3 * i + 2] = x;
        }
    }
}
