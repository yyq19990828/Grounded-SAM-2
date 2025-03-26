import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM

# 引入原始代码中的函数
from grounded_sam2_florence2_image_demo import (
    object_detection_and_segmentation,
    dense_region_caption_and_segmentation,
    region_proposal_and_segmentation,
    phrase_grounding_and_segmentation,
    referring_expression_segmentation,
    open_vocabulary_detection_and_segmentation,
    TASK_PROMPT
)

OUTPUT_DIR = "./outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化全局模型变量
florence2_model = None
florence2_processor = None
sam2_model = None
sam2_predictor = None

# 功能描述字典
PIPELINE_DESCRIPTIONS = {
    "object_detection_segmentation": {
        "name": "物体检测和分割",
        "description": "自动检测图像中的所有常见物体并生成分割掩码",
        "requires_text": False,
        "text_placeholder": "",
        "example_text": ""
    },
    "dense_region_caption_segmentation": {
        "name": "密集区域描述和分割",
        "description": "为图像中的不同区域生成详细描述并进行分割",
        "requires_text": False,
        "text_placeholder": "",
        "example_text": ""
    },
    "region_proposal_segmentation": {
        "name": "区域提议和分割",
        "description": "识别图像中的重要区域并进行分割",
        "requires_text": False,
        "text_placeholder": "",
        "example_text": ""
    },
    "phrase_grounding_segmentation": {
        "name": "短语定位和分割",
        "description": "根据输入的短语描述在图像中定位对象并分割",
        "requires_text": True,
        "text_placeholder": "输入需要定位的物体描述，例如：A green car, a yellow building",
        "example_text": "A green car, a yellow building"
    },
    "referring_expression_segmentation": {
        "name": "指代表达分割",
        "description": "根据指代表达式（如'图像中间的那只狗'）定位和分割对象",
        "requires_text": True,
        "text_placeholder": "输入指代表达，例如：the car in the middle of the image",
        "example_text": "the car in the middle of the image"
    },
    "open_vocabulary_detection_segmentation": {
        "name": "开放词汇检测和分割",
        "description": "根据用户提供的类别列表检测和分割图像中的对象",
        "requires_text": True,
        "text_placeholder": "输入需要检测的类别，多个类别用逗号分隔，例如：car, building, person",
        "example_text": "car, building, person"
    }
}

# 加载模型函数
def load_models():
    global florence2_model, florence2_processor, sam2_model, sam2_predictor
    
    if florence2_model is not None:
        return "模型已加载"
    
    FLORENCE2_MODEL_ID = "Florence-2-large-ft"
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 环境设置
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 加载 florence-2
    florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
    florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

    # 加载 sam 2
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    return "模型加载完成"

# 根据选择的管道更新界面
def update_interface(pipeline):
    pipeline_info = PIPELINE_DESCRIPTIONS[pipeline]
    requires_text = pipeline_info["requires_text"]
    text_placeholder = pipeline_info["text_placeholder"]
    example_text = pipeline_info["example_text"]
    description = pipeline_info["description"]
    
    return {
        text_input: gr.update(
            visible=requires_text, 
            placeholder=text_placeholder,
            value=example_text if requires_text else None,
            label=f"文本输入 {'(必填)' if requires_text else ''}"
        ),
        pipeline_description: gr.update(value=description)
    }

# 处理上传的图片和运行选择的管道
def process_image(image_path, pipeline, text_input=""):
    global florence2_model, florence2_processor, sam2_model, sam2_predictor
    
    if florence2_model is None:
        return "请先加载模型", None, None
    
    if PIPELINE_DESCRIPTIONS[pipeline]["requires_text"] and (text_input is None or text_input.strip() == ""):
        return f"错误: {PIPELINE_DESCRIPTIONS[pipeline]['name']}功能需要文本输入", None, None
    
    # 保存上传的图像
    temp_path = os.path.join(OUTPUT_DIR, "temp_input.jpg")
    if image_path is None:
        return "请先上传图像", None, None
        
    image = Image.fromarray(image_path).convert("RGB")
    image.save(temp_path)
    
    try:
        # 根据选择的管道运行相应的函数
        if pipeline == "object_detection_segmentation":
            object_detection_and_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=temp_path
            )
            box_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_det_annotated_image.jpg")
            mask_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_det_image_with_mask.jpg")
        
        elif pipeline == "dense_region_caption_segmentation":
            dense_region_caption_and_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=temp_path
            )
            box_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_dense_region_cap_annotated_image.jpg")
            mask_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_dense_region_cap_image_with_mask.jpg")
        
        elif pipeline == "region_proposal_segmentation":
            region_proposal_and_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=temp_path
            )
            box_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_region_proposal.jpg")
            mask_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_region_proposal_with_mask.jpg")
        
        elif pipeline == "phrase_grounding_segmentation":
            phrase_grounding_and_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=temp_path,
                text_input=text_input
            )
            box_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_phrase_grounding.jpg")
            mask_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_phrase_grounding_with_mask.jpg")
        
        elif pipeline == "referring_expression_segmentation":
            referring_expression_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=temp_path,
                text_input=text_input
            )
            box_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_referring_box.jpg")
            mask_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_referring_box_with_sam2_mask.jpg")
        
        elif pipeline == "open_vocabulary_detection_segmentation":
            open_vocabulary_detection_and_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=temp_path,
                text_input=text_input
            )
            box_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_open_vocabulary_detection.jpg")
            mask_path = os.path.join(OUTPUT_DIR, "grounded_sam2_florence2_open_vocabulary_detection_with_mask.jpg")
        
        # 读取结果图像
        box_result = cv2.imread(box_path)
        box_result = cv2.cvtColor(box_result, cv2.COLOR_BGR2RGB)
        
        mask_result = cv2.imread(mask_path)
        mask_result = cv2.cvtColor(mask_result, cv2.COLOR_BGR2RGB)
        
        return f"{PIPELINE_DESCRIPTIONS[pipeline]['name']}处理完成", box_result, mask_result
        
    except Exception as e:
        return f"处理过程中出错: {str(e)}", None, None

# 示例图像和示例函数
def load_example(example_idx):
    examples = {
        0: {
            "image": "notebooks/images/cars.jpg",
            "pipeline": "object_detection_segmentation",
            "text": ""
        },
        1: {
            "image": "notebooks/images/cars.jpg",
            "pipeline": "phrase_grounding_segmentation",
            "text": "A green car"
        },
        2: {
            "image": "notebooks/images/cars.jpg",
            "pipeline": "open_vocabulary_detection_segmentation", 
            "text": "car, wheel, door"
        }
    }
    
    example = examples.get(example_idx, examples[0])
    
    # 加载示例图像
    image_path = example["image"]
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = None
    
    # 获取所选功能的描述
    pipeline_value = example["pipeline"]
    description = PIPELINE_DESCRIPTIONS[pipeline_value]["description"]
    
    # 更新文本输入的可见性和内容
    text_visible = PIPELINE_DESCRIPTIONS[pipeline_value]["requires_text"]
    text_value = example["text"] if text_visible else ""
    
    # 返回列表而非字典，顺序与 outputs 参数中的组件顺序一致
    return img, pipeline_value, gr.update(visible=text_visible, value=text_value, 
                       label=f"文本输入 {'(必填)' if text_visible else ''}"), description

# 创建 Gradio 界面
def create_interface():
    global text_input, pipeline_description, image_input, pipeline
    
    with gr.Blocks(title="Grounded SAM 2 with Florence-2") as demo:
        gr.Markdown("# Grounded SAM 2 与 Florence-2 演示")
        
        with gr.Row():
            with gr.Column(scale=1):
                load_button = gr.Button("加载模型", variant="primary")
                status = gr.Textbox(label="状态", value="请先加载模型")
                load_button.click(fn=load_models, outputs=status)
                
                pipeline_choices = [(PIPELINE_DESCRIPTIONS[k]["name"], k) for k in PIPELINE_DESCRIPTIONS.keys()]
                pipeline = gr.Dropdown(
                    choices=pipeline_choices,
                    label="选择功能管道",
                    value="object_detection_segmentation"
                )
                
                pipeline_description = gr.Markdown(PIPELINE_DESCRIPTIONS["object_detection_segmentation"]["description"])
                
                text_input = gr.Textbox(
                    label="文本输入",
                    visible=False,
                    placeholder=PIPELINE_DESCRIPTIONS["phrase_grounding_segmentation"]["text_placeholder"],
                    info="根据所选功能输入相应的文本描述"
                )
                
                # 当管道选择改变时更新界面
                pipeline.change(
                    fn=update_interface, 
                    inputs=pipeline, 
                    outputs=[text_input, pipeline_description]
                )
                
                image_input = gr.Image(label="上传图像", type="numpy")
                run_button = gr.Button("运行", variant="primary")
                
                # 示例部分
                gr.Markdown("### 示例")
                with gr.Row():
                    example_button1 = gr.Button("示例1: 物体检测")
                    example_button2 = gr.Button("示例2: 短语定位")
                    example_button3 = gr.Button("示例3: 开放词汇检测")
                
                example_button1.click(fn=lambda: load_example(0), outputs=[image_input, pipeline, text_input, pipeline_description])
                example_button2.click(fn=lambda: load_example(1), outputs=[image_input, pipeline, text_input, pipeline_description]) 
                example_button3.click(fn=lambda: load_example(2), outputs=[image_input, pipeline, text_input, pipeline_description])
            
            with gr.Column(scale=2):
                with gr.Tab("边界框结果"):
                    box_output = gr.Image(label="边界框检测结果")
                with gr.Tab("分割掩码结果"):
                    mask_output = gr.Image(label="分割掩码结果")
        
        run_button.click(
            fn=process_image,
            inputs=[image_input, pipeline, text_input],
            outputs=[status, box_output, mask_output]
        )
        
        gr.Markdown("""
        ## 功能详细介绍
        
        ### 1. 物体检测和分割 (object_detection_segmentation)
        自动检测图像中的常见物体并生成分割掩码。不需要额外的文本输入。
        
        ### 2. 密集区域描述和分割 (dense_region_caption_segmentation)
        为图像中的区域生成详细描述并进行分割。系统会自动找出图像中的有意义区域并描述它们。
        
        ### 3. 区域提议和分割 (region_proposal_segmentation) 
        识别图像中的重要区域并进行分割，不提供具体的语义标签。
        
        ### 4. 短语定位和分割 (phrase_grounding_segmentation)
        根据提供的文本描述在图像中定位对应的物体并分割。需要输入文本描述，例如："A green car, a yellow building"。
        
        ### 5. 指代表达分割 (referring_expression_segmentation)
        通过指代表达式定位和分割特定物体。需要输入指代表达式，例如："the car in the middle of the image"。
        
        ### 6. 开放词汇检测和分割 (open_vocabulary_detection_segmentation)
        根据用户提供的类别列表检测和分割图像中的对象。需要输入类别列表，多个类别用逗号分隔，例如："car, building, person"。
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", share=True)
