import os
import folder_paths
import comfy.model_management as mm
import io
import base64
import torch
import requests
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import Union, List
from huggingface_hub import snapshot_download
import re
import numpy as np
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode  # 图像插值模式

# 添加类变量缓存模型
loaded_model = None
InternVL_model_name = None

class InternVLModelLoader:
    global loaded_model
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    [
                        "OpenGVLab/InternVL3-1B",
                        "OpenGVLab/InternVL3-2B",
                        "OpenGVLab/InternVL3-8B",
                    ],
                    {
                        "default": "OpenGVLab/InternVL3-2B"
                    }),
            }
        }

    RETURN_TYPES = ("InternVLModel",)
    RETURN_NAMES = ("intervl_model",)
    FUNCTION = "load_model"
    CATEGORY = "internvl"



    def load_model(self, model):
        global InternVL_model_name
        device = mm.get_torch_device()

        model_name = model.rsplit('/', 1)[-1]
        InternVL_model_name = model_name
        model_dir = (os.path.join(folder_paths.models_dir, "LLM", model_name))

        if not os.path.exists(model_dir):
            print(f"Downloading {model}")
            snapshot_download(repo_id=model, cache_dir=model_dir, local_dir_use_symlinks=False)
            # huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-2B --local-dir InternVL2-2B

        model = AutoModel.from_pretrained(
            model_dir,
            load_in_8bit=True,
            #load_in_4bit=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            #device_map="auto",
            trust_remote_code=True).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
        model = {
                    "model": model,
                    "tokenizer": tokenizer
                }
        InternVLModelLoader.loaded_model = model
        return (model,)


class DynamicPreprocess:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "min_num": ("INT", {"default": 1, "min": 1, "max": 40}),
                "max_num": ("INT", {"default": 12, "min": 1, "max": 40}),
                "image_size": ("INT", {"default": 448, }),
                "use_thumbnail": ("BOOLEAN", {"default": True, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "internvl"


    def load_image(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        #print("image:\n",image)
        pil_image = self.convert_to_pil_image(image)
        transform = self.build_transform(input_size=image_size)
        images = self.preprocess(pil_image, min_num, max_num, image_size, use_thumbnail)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return (pixel_values,)

    def preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # 生成所有可能的宽高比组合
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # 寻找与给定aspect_ratio最接近的宽高比
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))  # 裁剪子图

        # 可选添加缩略图
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def convert_to_pil_image(self, image: Union[
        np.ndarray, List[np.ndarray], bytes, str, Image.Image, torch.Tensor]) -> Image.Image:

        try:
            if isinstance(image, np.ndarray):
                print ("np.ndarray")
                return Image.fromarray(self._ensure_rgb(image))

            elif isinstance(image, list):
                print ("list")
                return self._handle_list_input(image)

            elif isinstance(image, bytes):
                print ("bytes")
                return Image.open(io.BytesIO(image)).convert('RGB')

            elif isinstance(image, str):
                print ("str")
                return self._handle_string_input(image)

            elif isinstance(image, Image.Image):
                print ("Image.Image")
                return image.convert('RGB')

            elif isinstance(image, torch.Tensor):
                print ("torch.Tensor")
                return self._convert_tensor_to_pil(image)

            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")

    def _handle_list_input(self, image_list: List) -> Image.Image:
        if len(image_list) == 0:
            raise ValueError("Empty list provided as image")

        if isinstance(image_list[0], np.ndarray):
            return Image.fromarray(self._ensure_rgb(image_list[0]))

        elif all(isinstance(x, (int, float)) for x in image_list):
            arr = np.array(image_list).astype('uint8')

            if arr.size in [1024 * 1024, 1024 * 1024 * 3]:
                arr = arr.reshape((1024, 1024, -1))
            elif arr.size in [512 * 512, 512 * 512 * 3]:
                arr = arr.reshape((512, 512, -1))
            else:
                arr = arr.reshape((arr.shape[0], -1))
            return Image.fromarray(self._ensure_rgb(arr))

        else:
            raise ValueError(f"Unsupported list content type: {type(image_list[0])}")

    def _handle_string_input(self, image_string: str) -> Image.Image:
        if image_string.startswith(('http://', 'https://')):
            response = requests.get(image_string)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')

        elif image_string.startswith('data:image'):
            image_data = base64.b64decode(image_string.split(',')[1])
            return Image.open(io.BytesIO(image_data)).convert('RGB')

        else:
            return Image.open(image_string).convert('RGB')

    def _ensure_rgb(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            return arr
        elif arr.ndim == 3 and arr.shape[2] == 4:
            return arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    def _convert_tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
        elif tensor.ndim == 2:

            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)

        np_array = tensor.cpu().numpy()

        if np_array.dtype != np.uint8:
            if np_array.max() <= 1.0:
                np_array = (np_array * 255).astype(np.uint8)
            else:
                np_array = np_array.astype(np.uint8)

        return Image.fromarray(self._ensure_rgb(np_array))

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

class InternVLHFInference:
    global loaded_model
    global InternVL_model_name

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "详细描述不要包含任何Markdown格式的图片标记，不需要图片缩略图片回复，直接输出文字描述内容。"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "请用一段自然语言的句子详细描述图片，所有物品需要明确数量，并说明相关场景元素位于场景图片的那个方位。"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": ("InternVLModel",),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "do_sample": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "internvl"

    def process(self,
                system_prompt,  # 必需参数第一位
                prompt,         # 必需参数第二位
                image=None,     # 可选参数，添加默认值
                model=None,     # 可选参数，添加默认值
                keep_model_loaded=False,
                max_new_tokens=1024,
                do_sample=False):
        
        mm.soft_empty_cache()
        device = mm.get_torch_device()

        # === 修复：安全的模型加载逻辑 ===
        using_cached_model = False

        #print("loaded_model:\n",InternVLModelLoader.loaded_model)
        print("keep_model_loaded:\n",keep_model_loaded)

        # 检查是否使用缓存模型
        if keep_model_loaded and InternVLModelLoader.loaded_model:
            model = InternVLModelLoader.loaded_model
            print("✅ 使用已缓存的模型")
            using_cached_model = True
            
        else:
            print("✅ 使用了传入的模型")
            
        # =============================
        
        model = InternVLModelLoader.loaded_model
        #print("model:\n",model)

        # 确保模型有效
        if not isinstance(model, dict) or 'model' not in model or model['model'] is None:
            # 需要加载新模型
            model_loader = InternVLModelLoader()
            model = model_loader.load_model(InternVL_model_name)[0]
            print("✅ 新模型加载成功")

            # keep_model_loaded = True
            # using_cached_model = True

            # # 缓存模型如果设置了保持加载
            # if keep_model_loaded:
            #     InternVLModelLoader.loaded_model = model
            #     print("✅ 模型已缓存供后续使用")
            #     keep_model_loaded = False

        internvl_model = model['model']
        tokenizer = model['tokenizer']
        
        #num_patches_list = [image.size(0)]

        # 准备流式输出
        streamer = TextIteratorStreamer(tokenizer, timeout=60)
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, streamer=streamer)


        # 处理图像输入
        if image is not None:
            image = image.to(torch.float16).to(device)
            num_patches_list = [image.size(0)]  # 有图像时设置

            question = f'<image>\n{system_prompt}\n{prompt}'

            # 启动生成线程
            Thread(target = internvl_model.chat, kwargs={
                "tokenizer": tokenizer,
                "pixel_values": image,  # 使用处理后的张量
                "question": question,
                "generation_config": generation_config,
                "num_patches_list": num_patches_list,
                "history": None,  # 关键修改：每次使用空历史
                "return_history": False  # 不再需要返回历史
            }).start()

        else:
            # 如果没有图像输入，创建一个空张量占位
            # image = torch.zeros(1, 3, 448, 448, dtype=torch.float16, device=device)
            # num_patches_list = [1]   # 无图像时设置

            question = f'{system_prompt}\n{prompt}'

            # 启动生成线程
            Thread(target = internvl_model.chat, kwargs={
                "tokenizer": tokenizer,
                "pixel_values": None,  # 使用处理后的张量
                "question": question,
                "generation_config": generation_config,
                "history": None,  # 关键修改：每次使用空历史
                "return_history": False  # 不再需要返回历史
            }).start()

        # 流式响应
        response = ''
        # Loop through the streamer to get the new text as it is generated
        for token in streamer:
            if token == internvl_model.conv_template.sep:
                continue
            print(token, end="\n", flush=True)  # Print each new chunk of generated text on the same line
            if "<|im_end|>" in token:
                token = token.replace("<|im_end|>", "")
                response += token
                if token:  # 如果删除后还有内容，继续发送剩余部分
                    continue
            response += token

        response = re.sub(r'!\[.*?\]\(.*?\)', '', response).strip()

        # === 修复：安全的模型卸载逻辑 ===
        if not keep_model_loaded and not using_cached_model:
            # 只卸载临时加载的模型
            print("♻️ 卸载临时模型...")
            
            # 释放模型资源
            if 'model' in model and model['model'] is not None:
                del model['model']
                model['model'] = None
            
            # 释放tokenizer资源
            if 'tokenizer' in model and model['tokenizer'] is not None:
                del model['tokenizer']
                model['tokenizer'] = None
            
            # 清除GPU缓存
            torch.cuda.empty_cache()
            print("✅ 临时模型已卸载")
        
        return (response,)
    


NODE_CLASS_MAPPINGS = {
    "InternVLModelLoader": InternVLModelLoader,
    "DynamicPreprocess": DynamicPreprocess,
    "InternVLHFInference": InternVLHFInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InternVLModelLoader": "InternVL Model Loader",
    "DynamicPreprocess": "Dynamic Preprocess",
    "InternVLHFInference": "InternVL HF Inference",
}