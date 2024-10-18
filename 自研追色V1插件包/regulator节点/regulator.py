from PIL import Image, ImageEnhance
import numpy as np
import torch

class tone_regulator_class:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "input_image": ("IMAGE",),
            },
            "optional": {
                "contrast_factor": ("FLOAT", {"default": 1.5, "min": 0.0, "step": 0.1}),
                "saturation_factor": ("FLOAT", {"default": 1.5, "min": 0.0, "step": 0.1}),
                "shadow_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "highlight_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "shadow_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "highlight_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1})
            }
        }

    CATEGORY = "plung-in/regulator"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    def enhance_image(self, image, contrast_factor, saturation_factor):
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        return image

    def extract_shadows_highlights(self, image, shadow_threshold, highlight_threshold):
        pixels = np.array(image)
        brightness = np.mean(pixels, axis=-1)

        shadows = pixels[brightness < shadow_threshold]
        highlights = pixels[brightness > highlight_threshold]

        shadow_color = np.mean(shadows, axis=0) if shadows.size else [0, 0, 0]
        highlight_color = np.mean(highlights, axis=0) if highlights.size else [255, 255, 255]

        return shadow_color, highlight_color

    def overlay_colors(self, base_image, shadow_color, highlight_color, shadow_alpha, highlight_alpha):
        # 创建阴影和高光图层
        shadow_layer = Image.new('RGB', base_image.size, tuple(map(int, shadow_color)))
        highlight_layer = Image.new('RGB', base_image.size, tuple(map(int, highlight_color)))

        # 混合图层
        shadow_layer = Image.blend(base_image, shadow_layer, shadow_alpha)
        result_image = Image.blend(base_image, highlight_layer, highlight_alpha)

        return Image.blend(shadow_layer, result_image, 0.5)

    def main(self, reference_image, input_image, contrast_factor, saturation_factor, shadow_threshold, highlight_threshold, shadow_alpha, highlight_alpha):
        i1 = input_image.squeeze(0).permute(0,1,2).mul(255).clamp(0,255).cpu().numpy().astype('uint8')
        image1 = Image.fromarray(i1, 'RGB')

        i2 = reference_image.squeeze(0).permute(0,1,2).mul(255).clamp(0,255).cpu().numpy().astype('uint8')
        image2 = Image.fromarray(i2, 'RGB')

        # 增强输入图像
        image1 = self.enhance_image(image1, contrast_factor, saturation_factor)
        
        shadow_color, highlight_color = self.extract_shadows_highlights(image2, shadow_threshold, highlight_threshold)
        
        image3 = self.overlay_colors(image1, shadow_color, highlight_color, shadow_alpha, highlight_alpha)

        _i = np.array(image3).astype(np.float32) / 255.0
        result_image = torch.from_numpy(_i)[None,]

        return result_image,

NODE_CLASS_MAPPINGS = {
    "regulator": tone_regulator_class,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "regulator": "tone-regulator",
}
