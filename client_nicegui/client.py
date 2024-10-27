
import websocket
import atexit
import asyncio

ws = websocket.create_connection("ws://localhost:8765")

@atexit.register
def close():
    ws.close()
    
from ngsolve import *
import base64, pickle, json

def Draw(obj1, *args, name=None, **kwargs):
    data = {}
    if isinstance(obj1, Mesh):
        d = pickle.dumps(obj1)
        data["type"] = "draw_mesh"
        
        data["data"] = { "mesh" : base64.b64encode(d).decode("utf-8") }
        if name is not None:
            data["data"]["name"] = name
    elif isinstance(obj1, GridFunction):
        print("draw gf")
        d = pickle.dumps(obj1)
        data["type"] = "draw_cf"
        data["data"] = { "cf" : base64.b64encode(d).decode("utf-8") }
        if name is not None:
            data["data"]["name"] = name
    elif isinstance(obj1, CoefficientFunction):
        print("draw cf")
        assert "mesh" in kwargs or len(args) > 0 and isinstance(args[0], Mesh)
        mesh = kwargs["mesh"] if "mesh" in kwargs else args[0]
        d = pickle.dumps({ "cf" : obj1, "mesh" : mesh })
        data["type"] = "draw_cf"
        data["data"] = { "objects" : base64.b64encode(d).decode("utf-8"),
                        "name" : name }
    else:
        raise ValueError("Unknown object type")
    ws.send(json.dumps(data))

two = False
m = Mesh(unit_square.GenerateMesh(maxh=0.2 if two else 0.1))
Draw(m)
cf = sin(10*x)
Draw(cf, m, name="sin10x")
Draw(cos(15*y), m, name="cos15y")