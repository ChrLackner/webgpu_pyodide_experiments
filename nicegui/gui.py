
from nicegui.element import Element
from webgpu_scene import WebGPUScene
from nicegui import app, ui
import ngsolve as ngs
import numpy as np


app.add_static_files("/pyodide", "../pyodide")
app.add_static_files("/webgpu", "../webgpu")
ui.add_head_html('<script type="text/javascript" src="./pyodide/pyodide.js"></script>')


class GUI(Element):
    def __init__(self):
        with ui.row():
            self.scene = WebGPUScene(width=1024, height=800)
            btn = ui.button("update")
        self.meshes = {}
        self.cfs = {}
        self.geometries = {}

    def register_handlers(self, message_handlers):
        self._message_handlers = message_handlers
        self._message_handlers["draw_mesh"] = self.draw_mesh

    def draw_mesh(self, data):
        import sys
        sys.path.append("../webgpu")
        from render_data import create_mesh_data
        import pickle
        import ngsolve as ngs
        import base64

        mesh = pickle.loads(base64.b64decode(data["mesh"].encode("utf-8")))
        name = data["name"] if "name" in data else "mesh" + str(len(self.meshes)+1)
        data = create_mesh_data(mesh)
        self.meshes[name] = data
        self.scene.draw_mesh(data)


        
