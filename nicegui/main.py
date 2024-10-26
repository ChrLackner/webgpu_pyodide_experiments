#!/usr/bin/env python3

from typing import Set
import asyncio
import json

from nicegui import app, ui
from nicegui.element import Element

import websockets
from websockets.server import WebSocketServerProtocol

import gui


CONNECTIONS: Set[WebSocketServerProtocol] = set()
message_handlers : dict[str, callable] = {}

async def handle_connect(websocket: WebSocketServerProtocol):
    """Register the new websocket connection, handle incoming messages and remove the connection when it is closed."""
    try:
        CONNECTIONS.add(websocket)
        async for data in websocket:
            data = json.loads(data)
            print("handle message", data, type(data))
            if data["type"] in message_handlers:
                message_handlers[data["type"]](data["data"])
    finally:
        CONNECTIONS.remove(websocket)

async def start_websocket_server():
    async with websockets.serve(handle_connect, 'localhost', 8765):
        await asyncio.Future()

app.on_startup(start_websocket_server)

g = gui.GUI()
g.register_handlers(message_handlers)

ui.run()


# inp = ui.input("function")
# button = ui.button("update")
# scene = WebGPUScene(width=1024, height=800)
# button.on_click(lambda _: scene.redraw(inp.value))


def draw_function(expr):
    import js
    import ngsolve as ngs
    import webgpu.main
    import webgpu.mesh
    from webgpu.main import gpu

    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.5))
    order = 6
    region = mesh.Region(ngs.VOL)
    cf = eval(expr, globals() | ngs.__dict__, locals())
    n_trigs, buffers = webgpu.mesh.create_mesh_buffers(gpu.device, region)
    buffers = buffers | webgpu.mesh.create_function_value_buffers(
        gpu.device, cf, region, order
    )
    mesh_object = webgpu.mesh.MeshRenderObject(gpu, buffers, n_trigs)
    webgpu.main.mesh_object = mesh_object
    js.requestAnimationFrame(webgpu.main.render_function)

