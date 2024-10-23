import js

from .uniforms import Binding
from .utils import SamplerBinding, TextureBinding, to_js


class Colormap:
    def __init__(self, device):
        self.device = device
        n = 5
        data = [0] * n * 4
        data[3::4] = [1] * n
        data[0 * 4 : 1 * 4 - 1] = [0, 0, 1]
        data[1 * 4 : 2 * 4 - 1] = [0, 1, 1]
        data[2 * 4 : 3 * 4 - 1] = [0, 1, 0]
        data[3 * 4 : 4 * 4 - 1] = [1, 1, 0]
        data[4 * 4 : 5 * 4 - 1] = [1, 0, 0]
        data = [255 * x for x in data]
        data = js.Uint8Array.new(data)

        self.texture = device.createTexture(
            to_js(
                {
                    "dimension": "1d",
                    "size": [n, 1, 1],
                    "format": "rgba8unorm",
                    "usage": js.GPUTextureUsage.TEXTURE_BINDING
                    | js.GPUTextureUsage.COPY_DST,
                }
            )
        )

        device.queue.writeTexture(
            to_js({"texture": self.texture}),
            data,
            to_js({"bytesPerRow": n * 4}),
            [n, 1, 1],
        )

        self.sampler = device.createSampler(
            to_js(
                {
                    "magFilter": "linear",
                    "minFilter": "linear",
                    "addressModeU": "clamp-to-edge",
                    "addressModeV": "clamp-to-edge",
                }
            )
        )

    def get_binding_layout(self, pipeline):
        FRAGMENT = js.GPUShaderStage.FRAGMENT
        return [
            {
                "binding": Binding.COLORMAP_TEXTURE,
                "visibility": FRAGMENT,
                "texture": {
                    "sampleType": "float",
                    "viewDimension": "1d",
                    "multisamples": False,
                },
            },
            {
                "binding": Binding.COLORMAP_SAMPLER,
                "visibility": FRAGMENT,
                "sampler": {"type": "filtering"},
            },
        ]

    def get_bindings(self):
        return [
            TextureBinding(Binding.COLORMAP_TEXTURE, self.texture),
            SamplerBinding(Binding.COLORMAP_SAMPLER, self.sampler),
        ]
        # return [
        #     {
        #         "binding": Binding.COLORMAP_TEXTURE,
        #         "resource": self.texture.createView(),
        #     },
        #     {"binding": Binding.COLORMAP_SAMPLER, "resource": self.sampler},
        # ]

    def __del__(self):
        self.texture.destroy()
