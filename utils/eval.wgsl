fn evalTrigP2Basis(lam: vec2<f32>) -> array<f32, 6> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x-y;
    let x0 = 2.0*x;
    return array(x*x, x0*y, y*y, x0*z, 2.0*y*z, z*z);
}

fn evalTrigP2(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP2Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset+0];
    for (var i: u32 = 1; i < 6; i++) {
        result += basis[i] * trig_function_values[offset+i*stride];
    }
    return result;
}

fn evalTrigP3Basis(lam: vec2<f32>) -> array<f32, 10> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x-y;
    let x0 = 3.0*x*x;
    let x1 = 3.0*y*y;
    let x2 = 3.0*z*z;
    return array(x*x*x, x0*y, x*x1, y*y*y, x0*z, 6.0*x*y*z, x1*z, x*x2, x2*y, z*z*z);
}

fn evalTrigP3(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP3Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset+0];
    for (var i: u32 = 1; i < 10; i++) {
        result += basis[i] * trig_function_values[offset+i*stride];
    }
    return result;
}

fn evalTrigP4Basis(lam: vec2<f32>) -> array<f32, 15> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x-y;
    let x0 = 4.0*x*x*x;
    let x1 = y*y;
    let x2 = x*x;
    let x3 = 6.0*x2;
    let x4 = 4.0*y*y*y;
    let x5 = 12.0*z;
    let x6 = z*z;
    let x7 = 4.0*z*z*z;
    return array(x*x*x*x, x0*y, x1*x3, x*x4, y*y*y*y, x0*z, x2*x5*y, x*x1*x5, x4*z, x3*x6, 12.0*x*x6*y, 6.0*x1*x6, x*x7, x7*y, z*z*z*z);
}

fn evalTrigP4(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP4Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset+0];
    for (var i: u32 = 1; i < 15; i++) {
        result += basis[i] * trig_function_values[offset+i*stride];
    }
    return result;
}

fn evalTrigP5Basis(lam: vec2<f32>) -> array<f32, 21> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x-y;
    let x0 = 5.0*x*x*x*x;
    let x1 = y*y;
    let x2 = x*x*x;
    let x3 = 10.0*x2;
    let x4 = x*x;
    let x5 = y*y*y;
    let x6 = 10.0*x5;
    let x7 = 5.0*y*y*y*y;
    let x8 = 20.0*z;
    let x9 = 30.0*x4;
    let x10 = z*z;
    let x11 = z*z*z;
    let x12 = 10.0*x11;
    let x13 = 5.0*z*z*z*z;
    return array(x*x*x*x*x, x0*y, x1*x3, x4*x6, x*x7, y*y*y*y*y, x0*z, x2*x8*y, x1*x9*z, x*x5*x8, x7*z, x10*x3, x10*x9*y, 30.0*x*x1*x10, x10*x6, x12*x4, 20.0*x*x11*y, x1*x12, x*x13, x13*y, z*z*z*z*z);
}

fn evalTrigP5(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP5Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset+0];
    for (var i: u32 = 1; i < 21; i++) {
        result += basis[i] * trig_function_values[offset+i*stride];
    }
    return result;
}

fn evalTrigP6Basis(lam: vec2<f32>) -> array<f32, 28> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x-y;
    let x0 = 6.0*x*x*x*x*x;
    let x1 = y*y;
    let x2 = x*x*x*x;
    let x3 = 15.0*x2;
    let x4 = y*y*y;
    let x5 = x*x*x;
    let x6 = 20.0*x5;
    let x7 = x*x;
    let x8 = y*y*y*y;
    let x9 = 15.0*x8;
    let x10 = 6.0*y*y*y*y*y;
    let x11 = 30.0*z;
    let x12 = 60.0*z;
    let x13 = z*z;
    let x14 = 60.0*x13;
    let x15 = z*z*z;
    let x16 = 60.0*x15;
    let x17 = z*z*z*z;
    let x18 = 15.0*x17;
    let x19 = 6.0*z*z*z*z*z;
    return array(x*x*x*x*x*x, x0*y, x1*x3, x4*x6, x7*x9, x*x10, y*y*y*y*y*y, x0*z, x11*x2*y, x1*x12*x5, x12*x4*x7, x*x11*x8, x10*z, x13*x3, x14*x5*y, 90.0*x1*x13*x7, x*x14*x4, x13*x9, x15*x6, x16*x7*y, x*x1*x16, 20.0*x15*x4, x18*x7, 30.0*x*x17*y, x1*x18, x*x19, x19*y, z*z*z*z*z*z);
}

fn evalTrigP6(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP6Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset+0];
    for (var i: u32 = 1; i < 28; i++) {
        result += basis[i] * trig_function_values[offset+i*stride];
    }
    return result;
}


fn evalTrig(id: u32, icomp: u32, lam: vec2<f32>) -> f32 {
    let order = u32(trig_function_values[1]);
    let ncomp = u32(trig_function_values[0]);
    let ndof = (order + 1) * (order + 2) / 2;

    let offset = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    if order == 1 { return evalTrigP1(offset, stride, lam); }
    if order == 2 { return evalTrigP2(offset, stride, lam); }
    if order == 3 { return evalTrigP3(offset, stride, lam); }
    if order == 4 { return evalTrigP4(offset, stride, lam); }
    if order == 5 { return evalTrigP5(offset, stride, lam); }
    if order == 6 { return evalTrigP6(offset, stride, lam); }

    return 0.0;
}
