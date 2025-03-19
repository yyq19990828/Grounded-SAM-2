# Grounded SAM 2：对视频中的任何事物进行 Grounding 和追踪

**[IDEA-Research](https://github.com/idea-research)**

[Tianhe Ren](https://rentainhe.github.io/), [Shuo Shen](https://github.com/ShuoShenDe)

[[`SAM 2 论文`](https://arxiv.org/abs/2408.00714)] [[`Grounding DINO 论文`](https://arxiv.org/abs/2303.05499)] [[`Grounding DINO 1.5 论文`](https://arxiv.org/abs/2405.10300)] [[`DINO-X 论文`](https://arxiv.org/abs/2411.14347)] [[`BibTeX`](#citation)]

[![视频名称](./assets/grounded_sam_2_intro.jpg)](https://github.com/user-attachments/assets/f0fb0022-779a-49fb-8f46-3a18a8b4e893)

## 亮点

Grounded SAM 2 是一个基础模型流水线，旨在通过 [Grounding DINO](https://arxiv.org/abs/2303.05499), [Grounding DINO 1.5](https://arxiv.org/abs/2405.10300), [Florence-2](https://arxiv.org/abs/2311.06242), [DINO-X](https://arxiv.org/abs/2411.14347) 和 [SAM 2](https://arxiv.org/abs/2408.00714) 对视频中的任何事物进行 grounding 和追踪。

在本仓库中，我们通过**简单的实现**支持了以下演示：
- 使用 Grounding DINO, Grounding DINO 1.5 & 1.6, DINO-X 和 SAM 2 **Grounding 和分割任何事物**
- 使用 Grounding DINO, Grounding DINO 1.5 & 1.6, DINO-X 和 SAM 2 **Grounding 和追踪任何事物**
- 基于强大的 [supervision](https://github.com/roboflow/supervision) 库的**检测、分割和追踪可视化**。

与 [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://arxiv.org/abs/2401.14159) 相比，Grounded SAM 2 没有引入重大的方法论上的改变。两种方法都利用了开放世界模型的能力来解决复杂的视觉任务。因此，我们尝试**简化代码实现**，以提高用户的便利性。

## 最新更新

- **2024.12.02**: 支持 **带有 SAM 2 的 DINO-X** 演示（包括对象分割和跟踪），请安装最新版本的 `dds-cloudapi-sdk==0.3.3` 并参考 [Grounded SAM 2 (with DINO-X)](#grounded-sam-2-image-demo-with-dino-x) 和 [Grounded SAM 2 Video (with DINO-X)](#grounded-sam-2-video-object-tracking-demo-with-custom-video-input-with-dino-x) 了解更多详情。

- **2024.10.24**: 支持 Grounded SAM 2 (使用 Grounding DINO 1.5) 上的 [SAHI (Slicing Aided Hyper Inference)](https://docs.ultralytics.com/guides/sahi-tiled-inference/)，这可能有助于推断具有密集小对象的高分辨率图像（例如 **4K** 图像）。

- **2024.10.10**: 支持 `SAM-2.1` 模型，如果要使用 `SAM 2.1` 模型，您需要更新到最新代码并重新安装 SAM 2，请按照 [SAM 2.1 安装](https://github.com/facebookresearch/sam2?tab=readme-ov-file#latest-updates) 进行操作。

- **2024.08.31**: 在 Grounded SAM 2 图像演示（使用 Grounding DINO）中支持 `dump json results`。

- **2024.08.20**: 支持 **Florence-2 SAM 2 图像演示**，其中包括 `dense region caption`、`object detection`、`phrase grounding` 和级联自动标注流水线 `caption + phrase grounding`。

- **2024.08.09**: 支持在整个视频中 **Grounding 和追踪新对象**。此功能目前仍在开发中。感谢 [Shuo Shen](https://github.com/ShuoShenDe)。

- **2024.08.07**: 支持 **自定义视频输入**，用户只需提交他们的视频文件（例如 `.mp4` 文件）和特定的文本提示，即可获得令人印象深刻的演示视频。

## 目录
- [Grounded SAM 2：对视频中的任何事物进行 Grounding 和追踪](#grounded-sam-2对视频中的任何事物进行-grounding-和追踪)
  - [亮点](#亮点)
  - [最新更新](#最新更新)
  - [目录](#目录)
  - [安装](#安装)
    - [无 Docker 安装](#无-docker-安装)
    - [使用 Docker 安装](#使用-docker-安装)
  - [Grounded SAM 2 演示](#grounded-sam-2-演示)
    - [Grounded SAM 2 图像演示（使用 Grounding DINO）](#grounded-sam-2-图像演示使用-grounding-dino)
    - [Grounded SAM 2 图像演示（使用 Grounding DINO 1.5 \& 1.6）](#grounded-sam-2-图像演示使用-grounding-dino-15--16)
    - [SAHI (Slicing Aided Hyper Inference) 与 Grounding DINO 1.5 和 SAM 2](#sahi-slicing-aided-hyper-inference-与-grounding-dino-15-和-sam-2)
    - [Grounded SAM 2 图像演示（使用 DINO-X）](#grounded-sam-2-图像演示使用-dino-x)
    - [自动保存 Grounding 结果（图像演示）](#自动保存-grounding-结果图像演示)
    - [Grounded SAM 2 视频对象追踪演示](#grounded-sam-2-视频对象追踪演示)
      - [支持各种追踪提示类型](#支持各种追踪提示类型)
    - [Grounded SAM 2 视频对象追踪演示（使用 Grounding DINO 1.5 \& 1.6）](#grounded-sam-2-视频对象追踪演示使用-grounding-dino-15--16)
    - [Grounded SAM 2 视频对象追踪演示，使用自定义视频输入（使用 Grounding DINO）](#grounded-sam-2-视频对象追踪演示使用自定义视频输入使用-grounding-dino)
    - [Grounded SAM 2 视频对象追踪演示，使用自定义视频输入（使用 Grounding DINO 1.5 \& 1.6）](#grounded-sam-2-视频对象追踪演示使用自定义视频输入使用-grounding-dino-15--16)
    - [Grounded SAM 2 视频对象追踪演示，使用自定义视频输入（使用 DINO-X）](#grounded-sam-2-视频对象追踪演示使用自定义视频输入使用-dino-x)
    - [Grounded-SAM-2 视频对象追踪，具有连续 ID（使用 Grounding DINO）](#grounded-sam-2-视频对象追踪具有连续-id使用-grounding-dino)
    - [Grounded-SAM-2 视频对象追踪，具有连续 ID 加上反向追踪（使用 Grounding DINO）](#grounded-sam-2-视频对象追踪具有连续-id-加上反向追踪使用-grounding-dino)
  - [Grounded SAM 2 Florence-2 演示](#grounded-sam-2-florence-2-演示)
    - [Grounded SAM 2 Florence-2 图像演示](#grounded-sam-2-florence-2-图像演示)
    - [Grounded SAM 2 Florence-2 图像自动标注演示](#grounded-sam-2-florence-2-图像自动标注演示)
    - [引用](#引用)

## 安装

下载预训练的 `SAM 2` 检查点：

```bash
cd checkpoints
bash download_ckpts.sh
```

下载预训练的 `Grounding DINO` 检查点：

```bash
cd gdino_checkpoints
bash download_ckpts.sh
```

### 无 Docker 安装

首先安装 PyTorch 环境。我们使用 `python=3.10`，以及 `torch >= 2.3.1`，`torchvision>=0.18.1` 和 `cuda-12.1` 在我们的环境中运行此演示。请按照[此处](https://pytorch.org/get-started/locally/)的说明安装 PyTorch 和 TorchVision 依赖项。强烈建议安装带有 CUDA 支持的 PyTorch 和 TorchVision。您可以轻松安装最新版本的 PyTorch，如下所示：

```bash
pip3 install torch torchvision torchaudio
```

由于我们需要 CUDA 编译环境来编译 Grounding DINO 中使用的 `Deformable Attention` 算子，因此我们需要检查 CUDA 环境变量是否已正确设置（您可以参考 [Grounding DINO 安装](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) 了解更多详细信息）。如果您想构建本地 GPU 环境以供 Grounding DINO 运行 Grounded SAM 2，您可以手动设置环境变量，如下所示：

```bash
export CUDA_HOME=/path/to/cuda-12.1/
```

安装 `Segment Anything 2`：

```bash
pip install -e .
```

安装 `Grounding DINO`：

```bash
pip install --no-build-isolation -e grounding_dino
```

### 使用 Docker 安装
构建 Docker 镜像并运行 Docker 容器：

```
cd Grounded-SAM-2
make build-image
make run
```
执行这些命令后，您将进入 Docker 环境。容器中的工作目录设置为：`/home/appuser/Grounded-SAM-2`

进入 Docker 环境后，您可以通过运行以下命令启动演示：
```
python grounded_sam2_tracking_demo.py
```

## Grounded SAM 2 演示
### Grounded SAM 2 图像演示（使用 Grounding DINO）
请注意，`Grounding DINO` 已经支持在 [Huggingface](https://huggingface.co/IDEA-Research/grounding-dino-tiny) 上使用，因此我们提供了两种运行 `Grounded SAM 2` 模型的方法：
- 使用 huggingface API 推理 Grounding DINO（简单明了）

```bash
python grounded_sam2_hf_model_demo.py
```

> [!NOTE]
> 🚨 如果在使用 `HuggingFace` 模型时遇到网络问题，您可以通过设置适当的镜像源来解决这些问题，例如 `export HF_ENDPOINT=https://hf-mirror.com`

- 加载本地预训练的 Grounding DINO 检查点，并使用 Grounding DINO 原始 API 进行推理（确保您已经下载了预训练的检查点）

```bash
python grounded_sam2_local_demo.py
```

### Grounded SAM 2 图像演示（使用 Grounding DINO 1.5 & 1.6）

我们已经发布了我们最强大的开放集检测模型 [Grounding DINO 1.5 & 1.6](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)，它可以与 SAM 2 结合使用，以获得更强大的开放集检测和分割能力。您可以首先申请 API 令牌，然后运行带有 Grounding DINO 1.5 的 Grounded SAM 2，如下所示：

安装最新的 DDS cloudapi：

```bash
pip install dds-cloudapi-sdk --upgrade
```

从我们的官方网站申请您的 API 令牌：[申请 API 令牌](https://deepdataspace.com/request_api)。

```bash
python grounded_sam2_gd1.5_demo.py
```

### SAHI (Slicing Aided Hyper Inference) 与 Grounding DINO 1.5 和 SAM 2

如果您的图像是具有密集对象的高分辨率图像，则直接使用 Grounding DINO 1.5 对原始图像进行推理可能不是最佳选择。我们支持 [SAHI (Slicing Aided Hyper Inference)](https://docs.ultralytics.com/guides/sahi-tiled-inference/)，它首先将原始图像分成更小的重叠块。然后分别对每个块执行推理，并合并最终的检测结果。这种方法对于高分辨率图像中密集和小对象的检测非常有效和准确。

您可以通过在 [grounded_sam2_gd1.5_demo.py](./grounded_sam2_gd1.5_demo.py) 中设置以下参数来运行 SAHI 推理：

```python
WITH_SLICE_INFERENCE = True
```

可视化效果如下所示：

| 文本提示 | 输入图像 | Grounded SAM 2 | 使用 SAHI 的 Grounded SAM 2 |
|:----:|:----:|:----:|:----:|
| `Person` | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam_2/demo_images/dense%20people.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam_2/grounding_dino_1.5_slice_inference/grounded_sam2_annotated_image_with_mask.jpg?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam_2/grounding_dino_1.5_slice_inference/grounded_sam2_annotated_image_with_mask_with_slice_inference.jpg?raw=true) |

- **注意：** 我们仅在 Grounding DINO 1.5 上支持 SAHI，因为它与更强大的 grounding 模型配合使用效果更好，从而产生更少的幻觉结果。

### Grounded SAM 2 图像演示（使用 DINO-X）

我们已经使用最强大的开放世界感知模型 [DINO-X](https://github.com/IDEA-Research/DINO-X-API) 实现了 Grounded SAM 2，以获得更好的开放集检测和分割性能。您可以首先申请 API 令牌，然后运行带有 DINO-X 的 Grounded SAM 2，如下所示：

安装最新的 DDS cloudapi：

```bash
pip install dds-cloudapi-sdk --upgrade
```

从我们的官方网站申请您的 API 令牌：[申请 API 令牌](https://deepdataspace.com/request_api)。

```bash
python grounded_sam2_dinox_demo.py
```

### 自动保存 Grounding 结果（图像演示）

在以下 Grounded SAM 2 图像演示中设置 `DUMP_JSON_RESULTS=True` 后：
- [grounded_sam2_local_demo.py](./grounded_sam2_local_demo.py)
- [grounded_sam2_hf_model_demo.py](./grounded_sam2_hf_model_demo.py)
- [grounded_sam2_gd1.5_demo.py](./grounded_sam2_gd1.5_demo.py)
- [grounded_sam2_dinox_demo.py](./grounded_sam2_dinox_demo.py)

`grounding` 和 `segmentation` 结果将自动保存在 `outputs` 目录中，格式如下：

```python
{
    "image_path": "path/to/image.jpg",
    "annotations": [
        {
            "class_name": "class_name",
            "bbox": [x1, y1, x2, y2],
            "segmentation": {
                "size": [h, w],
                "counts": "rle_encoded_mask"
            },
            "score": confidence score
        }
    ],
    "box_format": "xyxy",
    "img_width": w,
    "img_height": h
}
```

### Grounded SAM 2 视频对象追踪演示

基于 SAM 2 强大的追踪能力，我们可以将其与 Grounding DINO 结合使用，以进行开放集对象分割和追踪。您可以运行以下脚本来获取带有 Grounded SAM 2 的追踪结果：

```bash
python grounded_sam2_tracking_demo.py
```

- 每帧的追踪结果将保存在 `./tracking_results` 中
- 视频将保存为 `children_tracking_demo_video.mp4`
- 您可以使用不同的文本提示和视频片段自行优化此文件，以获得更多追踪结果。
- 为了简单起见，我们仅在此处使用 Grounding DINO 提示第一个视频帧。

#### 支持各种追踪提示类型

我们支持 Grounded SAM 2 追踪演示的不同类型的提示：

- **点提示**：为了**获得稳定的分割结果**，我们重新使用 SAM 2 图像预测器，以基于 Grounding DINO 框输出从每个对象获取预测掩码，然后我们**从预测掩码中均匀采样点**作为 SAM 2 视频预测器的点提示
- **框提示**：我们直接使用来自 Grounding DINO 的框输出作为 SAM 2 视频预测器的框提示
- **掩码提示**：我们使用基于 Grounding DINO 框输出的 SAM 2 掩码预测结果作为 SAM 2 视频预测器的掩码提示。

![Grounded SAM 2 追踪流水线](./assets/g_sam2_tracking_pipeline_vis_new.png)

### Grounded SAM 2 视频对象追踪演示（使用 Grounding DINO 1.5 & 1.6）

我们还支持基于我们更强大的 `Grounding DINO 1.5` 模型和 `SAM 2` 的视频对象追踪演示，您可以在申请 API 密钥以运行 `Grounding DINO 1.5` 后尝试以下演示：

```bash
python grounded_sam2_tracking_demo_with_gd1.5.py
```

### Grounded SAM 2 视频对象追踪演示，使用自定义视频输入（使用 Grounding DINO）

用户可以上传他们自己的视频文件（例如 `assets/hippopotamus.mp4`）并指定他们的自定义文本提示，以使用 Grounding DINO 和 SAM 2 进行 grounding 和追踪，方法是使用以下脚本：

```bash
python grounded_sam2_tracking_demo_custom_video_input_gd1.0_hf_model.py
```

如果您不方便使用 huggingface 演示，您也可以使用以下脚本通过本地 grounding dino 模型运行追踪演示：

```bash
python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py
```

### Grounded SAM 2 视频对象追踪演示，使用自定义视频输入（使用 Grounding DINO 1.5 & 1.6）

用户可以上传他们自己的视频文件（例如 `assets/hippopotamus.mp4`）并指定他们的自定义文本提示，以使用 Grounding DINO 1.5 和 SAM 2 进行 grounding 和追踪，方法是使用以下脚本：

```bash
python grounded_sam2_tracking_demo_custom_video_input_gd1.5.py
```

您可以在此文件中指定参数：

```python
VIDEO_PATH = "./assets/hippopotamus.mp4"
TEXT_PROMPT = "hippopotamus."
OUTPUT_VIDEO_PATH = "./hippopotamus_tracking_demo.mp4"
API_TOKEN_FOR_GD1_5 = "Your API token" # api token for G-DINO 1.5
PROMPT_TYPE_FOR_VIDEO = "mask" # 使用 SAM 2 掩码预测作为视频预测器的提示
```

运行我们的演示代码后，您可以获得如下所示的追踪结果：

[![视频名称](./assets/hippopotamus_seg.jpg)](https://github.com/user-attachments/assets/1fbdc6f4-3e50-4221-9600-98c397beecdf)

我们将自动将追踪可视化结果保存在 `OUTPUT_VIDEO_PATH` 中。

> [!WARNING]
> 我们在输入视频的第一帧上初始化框提示。如果您想从不同的帧开始，您可以在我们的代码中自行优化 `ann_frame_idx`。

### Grounded SAM 2 视频对象追踪演示，使用自定义视频输入（使用 DINO-X）

用户可以上传他们自己的视频文件（例如 `assets/hippopotamus.mp4`）并指定他们的自定义文本提示，以使用 DINO-X 和 SAM 2 进行 grounding 和追踪，方法是使用以下脚本：

```bash
python grounded_sam2_tracking_demo_custom_video_input_dinox.py
```

### Grounded-SAM-2 视频对象追踪，具有连续 ID（使用 Grounding DINO）

在上面的演示中，我们仅在特定帧中提示 Grounded SAM 2，这可能不利于在整个视频中查找新对象。在此演示中，我们尝试**查找新对象**并在整个视频中为它们分配新 ID，此功能**仍在开发中**，现在还不太稳定。

用户可以上传他们自己的视频文件，并使用 Grounding DINO 和 SAM 2 框架指定自定义文本提示以进行 grounding 和追踪。为此，请执行脚本：

```bash
python grounded_sam2_tracking_demo_with_continuous_id.py
```

您可以自定义各种参数，包括：

- `text`：grounding 文本提示。
- `video_dir`：包含视频文件的目录。
- `output_dir`：保存已处理输出的目录。
- `output_video_path`：输出视频的路径。
- `step`：用于处理的帧步长。
- `box_threshold`：groundingdino 模型的框阈值
- `text_threshold`：groundingdino 模型的文本阈值
注意：此方法仅支持掩码类型的文本提示。

运行我们的演示代码后，您可以获得如下所示的追踪结果：

[![视频名称](./assets/tracking_car_mask_1.jpg)](https://github.com/user-attachments/assets/d3f91ad0-3d32-43c4-a0dc-0bed661415f4)

如果您想尝试 `Grounding DINO 1.5` 模型，您可以在设置 API 令牌后运行以下脚本：

```bash
python grounded_sam2_tracking_demo_with_continuous_id_gd1.5.py
```

### Grounded-SAM-2 视频对象追踪，具有连续 ID 加上反向追踪（使用 Grounding DINO）
此方法可以简单地覆盖对象的整个生命周期
```bash
python grounded_sam2_tracking_demo_with_continuous_id_plus.py
```

## Grounded SAM 2 Florence-2 演示
### Grounded SAM 2 Florence-2 图像演示

在本节中，我们将探讨如何集成功能丰富且强大的开源模型 [Florence-2](https://arxiv.org/abs/2311.06242) 和 SAM 2 来开发实际应用。

[Florence-2](https://arxiv.org/abs/2311.06242) 是 Microsoft 的一个强大的视觉基础模型，它通过使用特殊的 `task_prompt` 进行提示来支持一系列视觉任务，包括但不限于：

| 任务 | 任务提示 | 文本输入 | 任务介绍 |
|:---:|:---:|:---:|:---:|
| 对象检测 | `<OD>` | &#10008; | 使用单个类别名称检测主要对象 |
| 密集区域描述 | `<DENSE_REGION_CAPTION>` | &#10008; | 使用简短描述检测主要对象 |
| 区域提议 | `<REGION_PROPOSAL>` | &#10008; | 生成没有类别名称的提议 |
| 短语 Grounding | `<CAPTION_TO_PHRASE_GROUNDING>` | &#10004; | Grounding 图像中标题中提到的主要对象 |
| 指代表达式分割 | `<REFERRING_EXPRESSION_SEGMENTATION>` | &#10004; | Grounding 与文本输入最相关的对象 |
| 开放词汇检测和分割 | `<OPEN_VOCABULARY_DETECTION>` | &#10004; | 使用文本输入 Grounding 任何对象 |

将 `Florence-2` 与 `SAM-2` 集成，我们可以构建一个强大的视觉流水线来解决复杂的视觉任务，您可以尝试以下脚本来运行演示：

> [!NOTE]
> 🚨 如果在使用 `HuggingFace` 模型时遇到网络问题，您可以通过设置适当的镜像源来解决这些问题，例如 `export HF_ENDPOINT=https://hf-mirror.com`

**对象检测和分割**
```bash
python grounded_sam2_florence2_image_demo.py \
    --pipeline object_detection_segmentation \
    --image_path ./notebooks/images/cars.jpg
```

**密集区域描述和分割**
```bash
python grounded_sam2_florence2_image_demo.py \
    --pipeline dense_region_caption_segmentation \
    --image_path ./notebooks/images/cars.jpg
```

**区域提议和分割**
```bash
python grounded_sam2_florence2_image_demo.py \
    --pipeline region_proposal_segmentation \
    --image_path ./notebooks/images/cars.jpg
```

**短语 Grounding 和分割**
```bash
python grounded_sam2_florence2_image_demo.py \
    --pipeline phrase_grounding_segmentation \
    --image_path ./notebooks/images/cars.jpg \
    --text_input "The image shows two vintage Chevrolet cars parked side by side, with one being a red convertible and the other a pink sedan, \
            set against the backdrop of an urban area with a multi-story building and trees. \
            The cars have Cuban license plates, indicating a location likely in Cuba."
```

**指代表达式分割**
```bash
python grounded_sam2_florence2_image_demo.py \
    --pipeline referring_expression_segmentation \
    --image_path ./notebooks/images/cars.jpg \
    --text_input "The left red car."
```

**开放词汇检测和分割**
```bash
python grounded_sam2_florence2_image_demo.py \
    --pipeline open_vocabulary_detection_segmentation \
    --image_path ./notebooks/images/cars.jpg \
    --text_input "car <and> building"
```
- 请注意，如果您想**检测多个类别**，您应该在输入文本中使用 `<and>` 分隔它们。

### Grounded SAM 2 Florence-2 图像自动标注演示
`Florence-2` 可以通过将其标题生成能力与其 grounding 能力级联使用，用作自动图像标注器。

| 任务 | 任务提示 | 文本输入 |
|:---:|:---:|:---:|
| 标题 + 短语 Grounding | `<CAPTION>` + `<CAPTION_TO_PHRASE_GROUNDING>` | &#10008; |
| 详细标题 + 短语 Grounding | `<DETAILED_CAPTION>` + `<CAPTION_TO_PHRASE_GROUNDING>` | &#10008; |
| 更详细的标题 + 短语 Grounding | `<MORE_DETAILED_CAPTION>` + `<CAPTION_TO_PHRASE_GROUNDING>` | &#10008; |

您可以尝试以下脚本来运行这些演示：

**标题到短语 Grounding**
```bash
python grounded_sam2_florence2_autolabel_pipeline.py \
    --image_path ./notebooks/images/groceries.jpg \
    --pipeline caption_to_phrase_grounding \
    --caption_type caption
```

- 您可以指定 `caption_type` 来控制标题的粒度，如果您想要更详细的标题，您可以尝试 `--caption_type detailed_caption` 或 `--caption_type more_detailed_caption`。

### 引用

如果您发现此项目对您的研究有帮助，请考虑引用以下 BibTeX 条目。

```BibTex
@misc{ravi2024sam2segmentimages,
      title={SAM 2: Segment Anything in Images and Videos}, 
      author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
      year={2024},
      eprint={2408.00714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00714}, 
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@misc{ren2024grounding,
      title={Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection}, 
      author={Tianhe Ren and Qing Jiang and Shilong Liu and Zhaoyang Zeng and Wenlong Liu and Han Gao and Hongjie Huang and Zhengyu Ma and Xiaoke Jiang and Yihao Chen and Yuda Xiong and Hao Zhang and Feng Li and Peijun Tang and Kent Yu and Lei Zhang},
      year={2024},
      eprint={2405.10300},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{ren2024grounded,
      title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks}, 
      author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
      year={2024},
      eprint={2401.14159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@misc{jiang2024trex2,
      title={T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy}, 
      author={Qing Jiang and Feng Li and Zhaoyang Zeng and Tianhe Ren and Shilong Liu and Lei Zhang},
      year={2024},
      eprint={2403.14610},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
