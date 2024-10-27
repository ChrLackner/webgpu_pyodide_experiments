
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
            self.scene = WebGPUScene(width=800, height=600)
            with ui.column():
                # use on function here to not be called when calling set_value
                self.obj_type = ui.toggle(["Geometry", "Mesh", "Solution"], value="Mesh").on('update:model-value', self.select_obj_type)
                self.selector = ui.select(options=[]).on('update:model-value', self.select_object)
        self.meshes = {}
        self.cfs = {}
        self.geometries = {}
        self.objects = { "Geometry" : self.geometries,
                         "Mesh" : self.meshes,
                         "Solution" : self.cfs
                        }
        self.last_object = { "Geometry" : None, "Mesh" : None, "Solution" : None }
        
    def select_obj_type(self, event):
        print(event)
        obj_type = event.args[1]["label"]
        self.selector.set_options(list(self.objects[obj_type].keys()))
        if self.selector.value not in self.objects[obj_type]:
            self.selector.set_value(self.last_object[obj_type])
            self.select_object(self.last_object[obj_type])
            
        

    def register_handlers(self, message_handlers):
        self._message_handlers = message_handlers
        self._message_handlers["draw_mesh"] = self.add_mesh
        self._message_handlers["draw_cf"] = self.add_cf
        
    def select_object(self, value):
        name = value if isinstance(value, str) else value.args["label"]
        self.last_object[self.obj_type.value] = name
        if self.obj_type.value == "Solution":
            self._draw_cf(**self.cfs[name])
        else:
            self._draw_mesh(self.meshes[name])

    def add_mesh(self, data):
        import pickle
        import ngsolve as ngs
        import base64

        mesh = pickle.loads(base64.b64decode(data["mesh"].encode("utf-8")))
        name = data["name"] if "name" in data else "mesh" + str(len(self.meshes)+1)
        self.meshes[name] = mesh
        if name not in self.selector.options:
            self.selector.set_options(list(self.meshes.keys()))
        self.last_object["Mesh"] = name
        self.selector.set_value(name)
        self._draw_mesh(mesh)
        
    def _draw_mesh(self, mesh):
        import sys
        sys.path.append("../webgpu")
        from render_data import create_mesh_data
        self.obj_type.set_value("Mesh")
        data = create_mesh_data(mesh)
        self.scene.draw_mesh(data)
        
    def add_cf(self, data):
        print("in add cf")
        import pickle
        import ngsolve as ngs
        import base64

        objects = pickle.loads(base64.b64decode(data["objects"].encode("utf-8")))
        name = data["name"]
        self.cfs[name] = objects
        self._change_obj_type("Solution")
        self.selector.set_value(name)
        self.last_object["Solution"] = name
        self._draw_cf(objects["cf"], objects["mesh"])
        
    def _change_obj_type(self, value):
        print("change obj types = ", value)
        print("current value = ")
        self.obj_type.set_value(value)
        print("selector options = ", self.objects[value].keys())
        self.selector.set_options(list(self.objects[value].keys()))
                
    def _draw_cf(self, cf, mesh):
        import sys
        sys.path.append("../webgpu")
        from render_data import create_cf_data
        data = create_cf_data(cf, mesh, order=2)
        self.scene.draw_cf(data)


        
