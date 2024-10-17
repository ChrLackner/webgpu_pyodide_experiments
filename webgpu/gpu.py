import js

from .colormap import Colormap
from .uniforms import Uniforms
from .utils import to_js


class WebGPU:
    """WebGPU management class, handles "global" state, like device, canvas, colormap and uniforms"""

    def __init__(self, device, canvas):
        self.render_function = None
        self.device = device
        self.format = js.navigator.gpu.getPreferredCanvasFormat()
        self.canvas = canvas

        self.uniforms = Uniforms(device)

        self.context = canvas.getContext("webgpu")
        self.context.configure(
            to_js(
                {
                    "device": device,
                    "format": self.format,
                    "alphaMode": "premultiplied",
                }
            )
        )
        self.colormap = Colormap(device)
        self.depth_format = "depth24plus"
        self.depth_stencil = {
            "format": self.depth_format,
            "depthWriteEnabled": True,
            "depthCompare": "less",
        }

        self.depth_texture = device.createTexture(
            to_js(
                {
                    "size": [canvas.width, canvas.height, 1],
                    "format": self.depth_format,
                    "usage": js.GPUTextureUsage.RENDER_ATTACHMENT,
                }
            )
        )

    def begin_render_pass(self, command_encoder):
        render_pass_encoder = command_encoder.beginRenderPass(
            to_js(
                {
                    "colorAttachments": [
                        {
                            "view": self.context.getCurrentTexture().createView(),
                            "clearValue": {"r": 1, "g": 1, "b": 1, "a": 1},
                            "loadOp": "clear",
                            "storeOp": "store",
                        }
                    ],
                    "depthStencilAttachment": {
                        "view": self.depth_texture.createView(
                            to_js({"format": self.depth_format, "aspect": "all"})
                        ),
                        "depthLoadOp": "clear",
                        "depthStoreOp": "store",
                        "depthClearValue": 1.0,
                    },
                },
            )
        )
        render_pass_encoder.setViewport(
            0, 0, self.canvas.width, self.canvas.height, 0.0, 1.0
        )
        return render_pass_encoder

    def __del__(self):
        self.depth_texture.destroy()
        del self.uniforms
        del self.colormap
