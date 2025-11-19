import gradio as gr
import torch
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ===== 配置路径 =====
config_file = "/mask2former_buaa.py" # 改成你的配置路径
checkpoint_file = "/buaa/实例分割/result_RGB/iter_11000.pth"  # 改成你的权重路径
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = init_detector(config_file, checkpoint_file, device=device)
print("Model loaded.")

# 类别名
CLASSES = model.dataset_meta['classes'] if hasattr(model, 'dataset_meta') else [str(i) for i in range(model.num_classes)]

def detect_and_debug(input_image, score_thr=0.5):
    results = inference_detector(model, input_image)
    # 提取预测结果
    if isinstance(results, tuple):
        results = results[0]
    pred_instances = results.pred_instances

    bboxes = getattr(pred_instances, "bboxes", None)
    labels = getattr(pred_instances, "labels", None)
    scores = getattr(pred_instances, "scores", None)
    masks_field = getattr(pred_instances, "masks", None)

    if bboxes is None or masks_field is None:
        return input_image, "⚠️ 没有检测到实例"

    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()
    # BitmapMasks
    if hasattr(masks_field, "to_ndarray"):
        masks = masks_field.to_ndarray()
    else:
        masks = masks_field.cpu().numpy()

    img_vis = input_image.copy()
    debug_lines = []

    for idx in range(len(scores)):
        if scores[idx] < score_thr:
            continue
        mask = masks[idx]

        # 转 bool (如果是float直接二值化)
        if mask.dtype != bool:
            mask_bin = mask > 0.5
        else:
            mask_bin = mask

        # 统计 mask 里 True 像素数量
        true_pixels = int(mask_bin.sum())

        # 随机颜色的透明叠加
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        img_vis[mask_bin] = img_vis[mask_bin] * 0.5 + color * 0.5

        class_name = CLASSES[labels[idx]] if labels[idx] < len(CLASSES) else str(labels[idx])
        # 这里只写调试信息，不画bbox
        debug_lines.append(
            f"ID {idx} | 类别: {class_name} | 分数: {scores[idx]:.3f} | mask像素: {true_pixels}"
        )

    # 拼接调试信息文本
    debug_info = "\n".join(debug_lines) if debug_lines else "⚠️ 没有符合阈值的实例"

    return img_vis, debug_info

# ===== Gradio 界面 =====
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="violet")) as demo:
    gr.HTML("""
    <div style="text-align:center; font-size: 28px; font-weight: bold; margin-bottom: 10px;">
        🎨 BUAA-人工智能原理与应用-4组 实例分割测试平台
    </div>
    <p style="text-align:center; font-size: 16px; color: #555;">
        上传图片 → 模型进行实例分割，并输出透明彩色 Mask 覆盖效果以及详细检测信息
    </p>
    <hr>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3>📤 上传图片</h3>")
            inp_image = gr.Image(label="点击或拖拽上传图片", type="numpy", show_download_button=True)
            score_slider = gr.Slider(label="分数阈值 🎯", minimum=0, maximum=1, value=0.5, step=0.05)
        with gr.Column(scale=1):
            gr.HTML("<h3>📷 分割可视化结果</h3>")
            out_image = gr.Image(label="预测结果", type="numpy", show_download_button=True)
            gr.HTML("<h3>📋 分割详细分析</h3>")
            debug_output = gr.Textbox(label="详细信息", lines=20)

    inp_image.change(fn=detect_and_debug, inputs=[inp_image, score_slider],
                     outputs=[out_image, debug_output])
    score_slider.change(fn=detect_and_debug, inputs=[inp_image, score_slider],
                        outputs=[out_image, debug_output])

demo.launch(server_name="0.0.0.0", server_port=7860)
