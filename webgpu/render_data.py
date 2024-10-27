
import ngsolve as ngs
import numpy as np
import math

def create_mesh_data(mesh):
    points = evaluate_cf(ngs.CF((ngs.x, ngs.y, ngs.z)), mesh.Region(ngs.VOL), order=1)

    n_trigs = len(mesh.ngmesh.Elements2D())

    edge_points = points[2:].reshape(-1, 3, 3)
    edges = np.zeros((n_trigs, 3, 2, 3), dtype=np.float32)
    for i in range(3):
        edges[:, i, 0, :] = edge_points[:, i, :]
        edges[:, i, 1, :] = edge_points[:, (i + 1) % 3, :]
    edge_data = edges.flatten().tobytes()
    
    trigs = np.zeros(
        n_trigs,
        dtype=[
            ("p", np.float32, 9),  # 3 vec3<f32> (each 4 floats due to padding)
            ("index", np.int32),  # index (i32)
        ],
    )
    trigs["p"] = points[2:].flatten().reshape(-1, 9)
    trigs["index"] = [1] * n_trigs
    return { "edges" : edge_data, "trigs" : trigs.tobytes(), "n_trigs" : n_trigs }

def create_cf_data(cf, mesh, order):
    data = create_mesh_data(mesh)
    data["cf"] = evaluate_cf(cf, mesh.Region(ngs.VOL), order).tobytes()
    return data

def evaluate_cf(cf, region, order):
    """Evaluate a coefficient function on a mesh and returns the values as a flat array, ready to copy to the GPU as storage buffer.
    The first two entries are the function dimension and the polynomial order of the stored values.
    """
    comps = cf.dim
    import ngsolve.webgui as ngwg
    int_points = ngwg._make_trig(order)
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

