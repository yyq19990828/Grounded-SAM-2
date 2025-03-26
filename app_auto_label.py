import os
import cv2
import torch
import gradio as gr
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM

# 定义常量和输出目录
OUTPUT_DIR = "./outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# 任务提示映射
TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
}

# 初始化模型
def initialize_models():
    print("正在加载模型...")
    # 环境设置
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 模型路径
    FLORENCE2_MODEL_ID = "Florence-2-large-ft"
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 构建 florence-2
    florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
    florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

    # 构建 sam 2
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    print("模型加载完成！")
    return florence2_model, florence2_processor, sam2_predictor

# 运行 florence2 模型的函数
def run_florence2(task_prompt, text_input, model, processor, image):
    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

# 处理图像的函数
def process_image(image, caption_type):
    if image is None:
        return "请上传图像", None, None
    
    # 保存上传的图像到临时文件
    temp_image_path = os.path.join(OUTPUT_DIR, "temp_input.jpg")
    image_pil = Image.fromarray(image).convert('RGB')
    image_pil.save(temp_image_path)
    
    # 获取描述任务提示
    caption_task_prompt = TASK_PROMPT[caption_type]
    
    # 获取图像描述
    caption_results = run_florence2(caption_task_prompt, None, florence2_model, florence2_processor, image_pil)
    text_input = caption_results[caption_task_prompt]
    result_text = f"图像描述: {text_input}"
    
    # 获取短语定位
    grounding_results = run_florence2(TASK_PROMPT["phrase_grounding"], text_input, florence2_model, florence2_processor, image_pil)
    grounding_results = grounding_results[TASK_PROMPT["phrase_grounding"]]
    
    # 解析 florence-2 检测结果
    input_boxes = np.array(grounding_results["bboxes"])
    class_names = grounding_results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # 使用 SAM 2 预测掩码
    sam2_predictor.set_image(np.array(image_pil))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # 指定标签
    labels = [f"{class_name}" for class_name in class_names]
    
    # 可视化结果
    img = np.array(image_pil)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    # 创建边界框标注
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    # 添加标签标注
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # 创建带掩码的标注
    mask_annotator = sv.MaskAnnotator()
    annotated_frame_with_mask = mask_annotator.annotate(scene=annotated_frame.copy(), detections=detections)
    
    return result_text, annotated_frame, annotated_frame_with_mask

# 初始化模型（可能需要一些时间）
florence2_model, florence2_processor, sam2_predictor = initialize_models()

# 创建 Gradio 界面
with gr.Blocks(title="Grounded SAM 2 + Florence 2 自动标注") as app:
    gr.Markdown("# Grounded SAM 2 + Florence 2 自动标注系统")
    gr.Markdown("上传一张图片，生成描述、短语定位和分割结果。")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图像", type="numpy")
            caption_type = gr.Radio(
                choices=["caption", "detailed_caption", "more_detailed_caption"], 
                label="描述类型", 
                value="caption",
                info="选择图像描述的详细程度"
            )
            run_button = gr.Button("处理图像", variant="primary")
        
        with gr.Column(scale=2):
            caption_output = gr.Textbox(label="生成的描述")
            with gr.Row():
                result_image = gr.Image(label="标注图像")
                result_image_mask = gr.Image(label="带掩码的标注图像")
    
    run_button.click(
        fn=process_image,
        inputs=[input_image, caption_type],
        outputs=[caption_output, result_image, result_image_mask]
    )

# 启动应用
if __name__ == "__main__":
    app.launch(share=True)
