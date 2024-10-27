
import websocket
import atexit
import asyncio

ws = websocket.create_connection("ws://localhost:8765")

@atexit.register
def close():
    ws.close()
    
from ngsolve import *
import base64, pickle, json

def Draw(mesh, name="mesh"):
    d = pickle.dumps(mesh)
    ws.send(json.dumps({ "type" : "draw_mesh", "data" : { "mesh" : base64.b64encode(d).decode("utf-8"),
                                                          "name" : name }}))
    print("sent mesh")

two = False
m = Mesh(unit_square.GenerateMesh(maxh=0.2 if two else 0.1))

Draw(m, name="mesh" + ("2" if two else ""))