# 在文件顶部添加
from server import PromptServer
from aiohttp import web  # 添加这行导入

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
from decord import VideoReader, cpu

# 添加类变量缓存模型
loaded_model = None
InternVL_model_name = None  # 初始化为None
InternVL_model_quantized = None  # 初始化为None

class hb_Number_Counter:
    def __init__(self):
        self.counters = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_type": (["integer", "float"],),
                "mode": (["increment", "decrement", "increment_to_restart", "decrement_to_restart"],),
                "start": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "stop": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "step": ("FLOAT", {"default": 1, "min": 0, "max": 99999, "step": 0.01}),
            },
            "optional": {
                "reset_bool": ("NUMBER",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("STRING", "float", "int")
    FUNCTION = "hb_increment_number"

    CATEGORY = "internvl"

    def hb_increment_number(self, number_type, mode, start, stop, step, unique_id, reset_bool=0):

        counter = int(start) if mode == 'integer' else start
        if self.counters.__contains__(unique_id):
            counter = self.counters[unique_id]

        if round(reset_bool) >= 1:
            counter = start

        # 根据模式更新计数器
        if mode == 'increment':
            counter += step
        elif mode == 'decrement':  # 修复了原代码中的拼写错误(deccrement)
            counter -= step
        elif mode == 'increment_to_restart':
            # 不满足条件时重置为start
            if counter < stop:
                counter += step
            else:
                counter = start
        elif mode == 'decrement_to_restart':
            # 不满足条件时重置为start
            if counter > stop:
                counter -= step
            else:
                counter = start

        self.counters[unique_id] = counter
        
        # 格式化结果
        hb_result = int(counter) if number_type == 'integer' else float(counter)
        # 注意：原代码中返回的字符串格式化可能不适用于浮点数，这里保持原样
        result = f"{hb_result:04d}" if number_type == 'integer' else str(hb_result)
        return (result, float(counter), int(counter))


class InternVLModelLoader:
    global loaded_model, InternVL_model_name, InternVL_model_quantized  # 声明使用全局变量
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    [
                        "OpenGVLab/InternVL3-1B-Instruct",
                        "OpenGVLab/InternVL3-2B-Instruct",
                        "OpenGVLab/InternVL3-8B-Instruct",
                        "OpenGVLab/InternVL3-14B-Instruct",
                    ],
                    {
                        "default": "OpenGVLab/InternVL3-8B-Instruct"
                    }),
                "quantized": (
                    [
                        "load_in_4bit",
                        "load_in_8bit",
                    ],
                    {
                        "default": "load_in_8bit"
                    }),

                    
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("intervl_model",)
    FUNCTION = "load_model"
    CATEGORY = "internvl"



    def load_model(self, model, quantized):
        global loaded_model, InternVL_model_name,InternVL_model_quantized  # 声明使用全局变量
        device = mm.get_torch_device()

        model_name = model.rsplit('/', 1)[-1]
        InternVL_model_name = model_name
        InternVL_model_quantized = quantized
        model_dir = (os.path.join(folder_paths.models_dir, "LLM", model_name))

        if not os.path.exists(model_dir):
            print(f"Downloading {model}")
            snapshot_download(repo_id=model, cache_dir=model_dir, local_dir_use_symlinks=False)
            # huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-2B --local-dir InternVL2-2B

        # 根据量化方式设置对应的布尔值
        if quantized == "load_in_4bit":
            load_in_4bit_boolean = True
            load_in_8bit_boolean = False
        elif quantized == "load_in_8bit":
            load_in_8bit_boolean = True
            load_in_4bit_boolean = False
        

        model = AutoModel.from_pretrained(
            model_dir,
            load_in_8bit=load_in_8bit_boolean,
            load_in_4bit=load_in_4bit_boolean,
            torch_dtype=torch.float16,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            #device_map="auto",
            trust_remote_code=True).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
        model_dict = {
            "model": model,
            "tokenizer": tokenizer
        }
        
        # 更新全局变量
        loaded_model = model_dict
        return (model_dict,)
    
    # 修改方法名为 ComfyUI 标准
    def HANDLE_CUSTOM_EVENT(self, node, event):
        global loaded_model, InternVL_model_name, InternVL_model_quantized
        if event.get("action") == "unload_model":
            if loaded_model is not None:
                print("开始卸载模型...")
                # 释放模型资源
                del loaded_model["model"]
                del loaded_model["tokenizer"]
                loaded_model = None
                # 不要重置模型名称，保留最后一次加载的模型名称
                
                # Cleanup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                print("✅ 模型已删除加载")
                return {"result": "success", "message": "模型已卸载"}
            else:
                return {"result": "error", "message": "没有加载的模型可卸载"}
        return {"result": "no_action"}  # 默认返回


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
    global loaded_model, InternVL_model_name, InternVL_model_quantized  # 声明使用全局变量
    video_IMAGENET_MEAN = (0.485, 0.456, 0.406)
    video_IMAGENET_STD = (0.229, 0.224, 0.225)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": """你是一位专业的AI图片描述工程师，擅长将图片转化为高质量、细节丰富、自然流畅的语言描述。

输出格式：
   - 使用自然流畅的语言描述，避免机械堆砌关键词
   - 按照视觉重要性排序元素，主体描述在前，环境氛围在后
   - 直接输出完整语言描述，不添加解释、分类标签或注释
   - 控制语言描述长度在适当范围内，确保核心元素突出

请直接返回自然流畅的语言描述，不需要解释你的思考过程或添加额外说明或任何Markdown格式的图片标记。
"""
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": """先描述摄影角度是怎样？然后详细描述场景图片的所有场景信息？所有场景元素需要明确数量，并说明相关场景元素位于图片的位置。包括所有人物的外貌、头发颜色、服装、姿态；所有场景元素的风格、颜色色调、材质、造型、装饰以及外观设计细节。
"""
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "model": ("MODEL",),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
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
                video_path=None,# 可选参数，添加默认值
                model=None,     # 可选参数，添加默认值
                keep_model_loaded=False,
                max_new_tokens=1024,
                do_sample=False):
        
        global loaded_model, InternVL_model_name, InternVL_model_quantized  # 声明使用全局变量
        
        mm.soft_empty_cache()
        device = mm.get_torch_device()

        # === 修复：安全的模型加载逻辑 ===
        using_cached_model = False

        print("keep_model_loaded:\n",keep_model_loaded)

        # 检查是否使用缓存模型
        if keep_model_loaded and loaded_model:  # 使用全局变量
            model = loaded_model
            print("✅ 使用已缓存的模型")
            using_cached_model = True
            
        # =============================

        # 确保模型有效
        if model is None or not isinstance(model, dict) or 'model' not in model or model['model'] is None:
            # 需要加载新模型
            model_loader = InternVLModelLoader()

            # 确保有有效的模型名称
            current_model_name = InternVL_model_name if InternVL_model_name else "OpenGVLab/InternVL3-14B-Instruct"
            current_model_quantized = InternVL_model_quantized if InternVL_model_quantized else "load_in_4bit"

            model = model_loader.load_model(InternVL_model_name,InternVL_model_quantized)[0]
            print(f"✅ 新模型加载成功: {current_model_name}")
            print(f"✅ 量化加载成功: {current_model_quantized}")


            # 缓存模型如果设置了保持加载
            if keep_model_loaded:
                loaded_model = model  # 使用全局变量
                print("✅ 模型已缓存供后续使用")

        internvl_model = model['model']
        tokenizer = model['tokenizer']
        
        # 准备流式输出
        streamer = TextIteratorStreamer(tokenizer, timeout=60)
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, streamer=streamer)

        # 处理视频输入
        if video_path and video_path.strip():
            print("处理视频输入")
            
            # 加载并处理视频（设置8个时间段，每个时间取1个分块）
            pixel_values, num_patches_list = self.video_load_video(video_path, num_segments=8, max_num=1)

            # 确保数据在GPU上
            if torch.cuda.is_available():
                # 将数据转换为bfloat16格式并转移到GPU
                pixel_values = pixel_values.to(torch.float16).cuda()
                
            # 构造视频前缀：为每个帧生成"FrameX: <image>\n"格式的文本
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = f'{video_prefix}\n{system_prompt}\n{prompt}'

            # 启动生成线程
            Thread(target = internvl_model.chat, kwargs={
                "tokenizer": tokenizer,
                "pixel_values": pixel_values,  # 使用处理后的张量
                "question": question,
                "generation_config": generation_config,
                "num_patches_list": num_patches_list,
                "history": None,  # 关键修改：每次使用空历史
                "return_history": False  # 不再需要返回历史
            }).start()


        # 处理图像输入
        elif image is not None:
            print("处理图像输入")
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
            print("纯文本推理")
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
            
            import gc
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            print("✅ 临时模型已卸载")
        
        return (response,)
    

    # 定义函数：根据时间范围和视频参数计算需要抽取的帧索引
    def video_get_index(self,bound, fps, max_frame, first_idx=0, num_segments=32):
        """参数说明：
        bound: 时间范围元组(start_sec, end_sec)
        fps: 视频帧率（帧/秒）
        max_frame: 视频总帧数
        first_idx: 起始帧索引（默认0）
        num_segments: 需要分割的视频段数（默认32）"""
        
        # 处理时间边界
        if bound:  # 如果指定了时间范围
            start, end = bound[0], bound[1]  # 获取开始和结束时间（秒）
        else:  # 未指定则使用极大范围
            start, end = -100000, 100000
        
        # 计算起始和结束帧索引
        start_idx = max(first_idx, round(start * fps))  # 转换为帧索引，确保不小于first_idx
        end_idx = min(round(end * fps), max_frame)  # 结束帧不超过视频最大帧
        
        # 计算每个视频段的长度（以帧为单位）
        seg_size = float(end_idx - start_idx) / num_segments
        
        # 生成每个视频段的中心帧索引
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices  # 返回32个均匀分布的帧索引


    # 定义视频加载和处理函数
    def video_load_video(self,video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        """参数说明：
        video_path: 视频文件路径
        bound: 时间范围（秒）
        input_size: 输入图像尺寸（默认448x448）
        max_num: 最大分块数量（动态预处理用）
        num_segments: 视频分割段数"""
        
        # 初始化视频阅读器
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)  # 使用CPU单线程读取
        max_frame = len(vr) - 1  # 获取视频总帧数（索引从0开始）
        fps = float(vr.get_avg_fps())  # 获取视频平均帧率

        # 初始化存储容器
        pixel_values_list = []  # 存储处理后的图像张量
        num_patches_list = []   # 存储每帧的分块数量
        
        # 创建图像预处理流水线
        transform = self.video_build_transform(input_size=input_size)  # 包含缩放、归一化等操作
        
        # 获取需要处理的帧索引
        frame_indices = self.video_get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        
        # 遍历每个选定帧进行处理
        for frame_index in frame_indices:
            # 读取帧并转换为PIL图像
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            
            # 动态预处理（可能包含分块、缩略图处理等）
            img = self.video_preprocess(img, 
                                    image_size=input_size,
                                    use_thumbnail=True,
                                    max_num=max_num)
            
            # 对每个分块应用预处理
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)  # 堆叠分块张量
            
            # 记录分块数量和预处理结果
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        # 合并所有帧的分块数据
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    def video_build_transform(self, input_size):
        MEAN, STD = self.video_IMAGENET_MEAN, self.video_IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def video_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # 生成所有可能的宽高比组合
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # 寻找与给定aspect_ratio最接近的宽高比
        target_aspect_ratio = self.video_find_closest_aspect_ratio(
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

    def video_find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
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
    


NODE_CLASS_MAPPINGS = {
    "InternVLModelLoader": InternVLModelLoader,
    "DynamicPreprocess": DynamicPreprocess,
    "InternVLHFInference": InternVLHFInference,
    "hb_Number_Counter": hb_Number_Counter,
}
#class要一致

NODE_DISPLAY_NAME_MAPPINGS = {
    "InternVLModelLoader": "InternVL Model Loader",
    "DynamicPreprocess": "Dynamic Preprocess",
    "InternVLHFInference": "InternVL HF Inference",
    "hb_Number_Counter": "计数器",
}

# 修改文件底部的路由处理函数
@PromptServer.instance.routes.post("/object_info/{node_class}")
async def handle_custom_node_event(request):
    try:
        post_data = await request.json()
        node_class = request.match_info.get("node_class", "")
        action = post_data.get("action")
        node_id = post_data.get("node_id")
        
        # 创建模型加载器实例
        loader = InternVLModelLoader()
        
        if node_class == "InternVLModelLoader" and action == "unload_model":
            # 调用模型卸载逻辑
            result = loader.HANDLE_CUSTOM_EVENT(None, {"action": "unload_model"})
            return web.json_response(result)
        else:
            return web.json_response(
                {"error": "Invalid action or node class"}, status=400
            )
    
    except Exception as e:
        return web.json_response(
            {"error": str(e)}, status=500
        )