import js

from .utils import create_proxy


class InputHandler:
    def __init__(self, canvas, uniforms, render_function=None):
        self.canvas = canvas
        self.uniforms = uniforms
        self.render_function = render_function
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
            self.uniforms.mat[12] += ev.movementX / self.canvas.width * 1.8
            self.uniforms.mat[13] -= ev.movementY / self.canvas.height * 1.8
            if self.render_function:
                js.requestAnimationFrame(self.render_function)

    def unregister_callbacks(self):
        for event in self._callbacks:
            for func in self._callbacks[event]:
                self.canvas.removeEventListener(event, func)
                func.destroy()
        self._callbacks = {}

    def on(self, event, func):
        func = create_proxy(func)
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(func)
        self.canvas.addEventListener(event, func)

    def register_callbacks(self):
        self.unregister_callbacks()
        self.on("mousedown", self.on_mousedown)
        self.on("mouseup", self.on_mouseup)
        self.on("mousemove", self.on_mousemove)

    def __del__(self):
        self.unregister_callbacks()
        if self.render_function:
            js.cancelAnimationFrame(self.render_function)
            self.render_function.destroy()
            self.render_function = None
