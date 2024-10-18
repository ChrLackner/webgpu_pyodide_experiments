import math

import ngsolve as ngs
from ngsolve.fem import ET
from sympy import *
from sympy.codegen.ast import float32, real
from sympy.printing.codeprinter import ccode
from sympy.simplify.cse_main import cse


def getReferenceRules(order, sd):
    n = S(order) * S(sd + 1) + 1
    h = S(1) / (n - 1)
    res = {}
    res[ngs.ET.SEGM] = [(i * h, S(0), S(0)) for i in range(n)]
    res[ngs.ET.TRIG] = [
        (i * h, j * h, 1 - i * h - j * h) for i in range(n) for j in range(n - i)
    ]
    res[ngs.ET.QUAD] = [(i * h, j * h, S(0)) for j in range(n) for i in range(n)]
    res[ngs.ET.TET] = [
        (i * h, j * h, k * h)
        for k in range(n)
        for j in range(n - k)
        for i in range(n - k - j)
    ]
    res[ngs.ET.HEX] = [
        (i * h, j * h, k * h) for k in range(n) for j in range(n) for i in range(n)
    ]
    res[ngs.ET.PRISM] = [
        (i * h, j * h, k * h) for k in range(n) for j in range(n) for i in range(n - j)
    ]

    # no high order pyramids
    n = S(2)
    h = S(1) / (n - 1)
    res[ngs.ET.PYRAMID] = [
        (i * h, j * h, k * h)
        for k in range(n)
        for j in range(n - k)
        for i in range(n - k)
    ]
    return res


_eval_template = """
fn eval{eltype}{suffix}(id: u32, icomp: u32, lam: {lam_type}) -> f32 {{
    let order: u32 = u32(trig_function_values[1]);
    let ncomp: u32 = u32(trig_function_values[0]);
    let ndof: u32 = {ndof_expr};

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

{switch_order}
    return 0.0;
}}
"""


def getTrigLagrangeBasisFunction(p, lam, x, i):
    numerator = 1.0
    denom = 1.0
    for k in range(p + 1):
        if i != k:
            numerator *= lam - x[k]
            denom *= x[i] - x[k]
    return numerator / denom


def Bernstein(x, y, z, i, j, n):
    fac = math.factorial
    return fac(n) / fac(i) / fac(j) / fac(n - i - j) * x ** (n - i - j) * y**i * z**j


def getBernsteinPolynomial(et, p, i, j=0, k=0):
    def Binomial(i):
        return math.factorial(p) / math.factorial(i) / math.factorial(p - i)

    def Bernstein(x, i):
        return Binomial(i) * x**i * (1 - x) ** (p - i)

    def phi(x, y, z):
        return Bernstein(x, j, og)

    for i in range(og + 1):
        for j in range(og + 1):
            Bvals[i, j] = Bernstein(i / og, j, og)
    iBvals = Bvals.I
    return iBvals


def getBasisFunction(et, p, i, j=0, k=0):
    # print("get", et, p, i, j, k)
    if et == ET.SEGM:

        def phi(x, y, z):
            return Bernstein(x, y, z, i, j, p)

    elif et == ET.TRIG:

        def phi(x, y, z):
            return Bernstein(x, y, z, i, j, p)

        # def phi(x,y,z):
        #     return  x**i*y**j
    elif et == ET.QUAD:

        def phi(x, y, z):
            return x**i * y**j

    elif et == ET.TET:

        def phi(x, y, z):
            return x**i * y**j * z**k

    elif et == ET.HEX:

        def phi(x, y, z):
            return x**i * y**j * z**k

    elif et == ET.PRISM:

        def phi(x, y, z):
            return x**i * y**j * z**k

    elif et == ET.PYRAMID:

        def phi(x, y, z):
            return x**i * y**j * z**k

    else:
        raise RuntimeError("unknown type: " + str(et))

    phi.i = i
    phi.j = j
    phi.k = k
    return phi


def getBasisFunctions(et, p):
    if et == ET.SEGM:
        return [getBasisFunction(et, p, i) for i in range(p + 1)]
    if et == ET.TRIG:
        return [
            getBasisFunction(et, p, i, j)
            for j in range(p + 1)
            for i in range(p + 1 - j)
        ]
    if et == ET.QUAD:
        return [
            getBasisFunction(et, p, i, j) for i in range(p + 1) for j in range(p + 1)
        ]
    if et == ET.TET:
        return [
            getBasisFunction(et, p, i, j, k)
            for i in range(p + 1)
            for j in range(p + 1 - i)
            for k in range(p + 1 - i - j)
        ]
    if et == ET.HEX:
        return [
            getBasisFunction(et, p, i, j, k)
            for i in range(p + 1)
            for j in range(p + 1)
            for k in range(p + 1)
        ]
    if et == ET.PRISM:
        return [
            getBasisFunction(et, p, i, j, k)
            for k in range(p + 1)
            for j in range(p + 1)
            for i in range(p + 1 - j)
        ]
    if et == ET.PYRAMID:
        return [
            getBasisFunction(et, p, i, j, k)
            for k in range(1 + 1)
            for j in range(1 + 1 - k)
            for i in range(1 + 1 - k)
        ]


def GenerateInterpolationFunction(et, orders, scal_dims):
    dim = {
        ET.SEGM: 1,
        ET.TRIG: 2,
        ET.TET: 3,
        ET.QUAD: 2,
        ET.HEX: 3,
        ET.PRISM: 3,
        ET.PYRAMID: 3,
    }[et]
    eltype = {
        ET.SEGM: "Seg",
        ET.TRIG: "Trig",
        ET.TET: "Tet",
        ET.QUAD: "Quad",
        ET.HEX: "Hex",
        ET.PRISM: "Prism",
        ET.PYRAMID: "Pyramid",
    }[et]
    ndof_expr = {
        ET.SEGM: "order+1",
        ET.TRIG: "(order+1)*(order+2)/2",
        ET.TET: "(order+1)*(order+2)*(order+3)/6",
        ET.QUAD: "(order+1)*(order+1)",
        ET.HEX: "(order+1)*(order+1)*(order+1)",
        ET.PRISM: "(order+1)*(order+1)*(order+2)/2",
        ET.PYRAMID: "",
    }[et]
    result = ""
    for p in orders:
        print("\n\n=============", eltype, p)
        ir = getReferenceRules(p, 0)[et]
        basis = getBasisFunctions(et, p)
        ndof = len(basis)
        nips = len(ir)
        if nips != ndof:
            raise RuntimeError(
                "Number of ips ({}) and dofs ({}) doesn't match for element {} and order {}".format(
                    nips, ndof, et, p
                )
            )

        u, v, w = symbols("x y z")
        basis_exprs = [phi(u, v, w) for phi in basis]
        vars, exprs = cse(basis_exprs, optimizations="basic")

        def fix_pow(code):
            while "powf" in code:
                start = code.find("powf")
                end = code.find(")", start)
                pow_str = code[start : end + 1]
                # convert pow(x, 2) to x*x
                var, exp = pow_str[5:-1].split(",")
                exp = int(exp)
                new_pow_str = "*".join([var] * exp)
                # print("replacing", pow_str, "with", new_pow_str)
                code = code.replace(pow_str, new_pow_str)
            return code

        def to_ccode(code):
            code = ccode(code.evalf(), type_aliases={real: float32}).replace("F", "")
            if code.startswith("1.0*"):
                code = code[4:]
            return fix_pow(code)

        lam_type = f"vec{dim}<f32>" if dim > 1 else "f32"
        lam_name = "x" if dim == 1 else "lam"
        basis_code = f"fn eval{eltype}P{p}Basis({lam_name}: {lam_type}) -> array<f32, {ndof}> {{\n"
        names = ["x", "y", "z", "w"]
        if dim > 1:
            for i in range(dim):
                basis_code += f"    let {names[i]} = lam.{names[i]};\n"
        basis_code += f"    let {names[dim]} = 1.0 - {'-'.join(names[:dim])};\n"

        for name, value in vars:
            code = to_ccode(value)
            basis_code += f"    let {name} = {code};\n"

        basis_code += (
            "    return array(" + ", ".join([to_ccode(expr) for expr in exprs]) + ");\n"
        )
        basis_code += "}\n\n"
        print(ndof + basis_code.count("*"), "multiplications")
        print(ndof + basis_code.count("+"), "additions")

        def code_get_vec(dim, i=0):
            if dim == 1:
                return f"{eltype.lower()}_function_values[offset+{i}*stride]"
            code = f"vec{dim}<f32>("
            for i in range(dim):
                code += f"{eltype.lower()}_function_values[offset+{i}*stride]"
                if i < dim - 1:
                    code += ", "
            code += ")"
            return code

        code = basis_code
        for scal_dim in scal_dims:
            if scal_dim == 1:
                scal = "f32"
                suffix = ""
            else:
                scal = f"vec{scal_dim}<f32>"
                suffix = f"V{scal_dim}"

            code += f"fn eval{eltype}P{p}{suffix}(offset: u32, stride: u32, lam: {lam_type}) -> {scal} {{\n"
            code += f"    let basis = eval{eltype}P{p}Basis(lam);\n"
            code += f"    var result: {scal} = basis[0] * {code_get_vec(scal_dim)};\n"
            for i in range(1, ndof):
                code += f"    result += basis[{i}] * {code_get_vec(scal_dim, i)};\n"
            code += f"    return result;\n"
            code += f"}}\n\n"
        result += code

    for scal_dim in scal_dims:
        if scal_dim == 1:
            scal = "f32"
            suffix = ""
        else:
            scal = f"vec{scal_dim}<f32>"
            suffix = f"V{scal_dim}"
        switch_order = ""
        orders_ = sorted(list(set(list(orders) + [1])))
        for p in orders_:
            switch_order += f"    if order == {p} {{ return eval{eltype}P{p}{suffix}(offset, stride, lam); }}\n"
        result += _eval_template.format(**locals())
    return result


code = ""
for et in [ET.SEGM, ET.TRIG, ET.TET][0:2]:
    code += GenerateInterpolationFunction(et, orders=range(1, 7), scal_dims=range(1, 2))

open("../webgpu/eval.wgsl", "w").write(code)
