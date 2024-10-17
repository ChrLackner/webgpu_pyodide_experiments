import js

from .utils import create_proxy


class InputHandler:
    def __init__(self, gpu):
        self.gpu = gpu
        self._is_moving = False

        self._callbacks = {}
        self.register_callbacks()

    def on_mousedown(self, _):
        self._is_moving = True

    def on_mouseup(self, _):
        global _is_moving
        self._is_moving = False

    def on_mousemove(self, ev):
        if self._is_moving:
            self.gpu.uniforms.mat[12] += ev.movementX / self.gpu.canvas.width * 1.8
            self.gpu.uniforms.mat[13] -= ev.movementY / self.gpu.canvas.height * 1.8
            js.requestAnimationFrame(self.gpu.render_function)

    def unregister_callbacks(self):
        for event in self._callbacks:
            for func in self._callbacks[event]:
                self.gpu.canvas.removeEventListener(event, func)
                func.destroy()
        self._callbacks = {}

    def on(self, event, func):
        func = create_proxy(func)
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(func)
        self.gpu.canvas.addEventListener(event, func)

    def register_callbacks(self):
        self.unregister_callbacks()
        self.on("mousedown", self.on_mousedown)
        self.on("mouseup", self.on_mouseup)
        self.on("mousemove", self.on_mousemove)

    def __del__(self):
        self.unregister_callbacks()
