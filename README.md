# MATHGLANCE: Evaluating Visual Perception in Mathematical Visual Contexts 

![MathQA](https://img.shields.io/badge/Task-MathQA-red) 
![Mathematical Reasoning](https://img.shields.io/badge/Task-Mathematical_Reasoning-red) 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 

Official repository for the paper "[MATHGLANCE: Evaluating Visual Perception in Mathematical Visual Contexts](https://arxiv.org/pdf/2503.20745)".

🌟 For more details, please refer to the project page with benchmark overview: [https://mathglance.github.io/](https://mathglance.github.io/).

[[🌐 Webpage](ttps://mathglance.github.io/)] [[📖 Paper](https://arxiv.org/pdf/2503.2074)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/zs0506/GeoPeP_Caption)] [[🤗 Checkpoints](https://huggingface.co/zs0506/SVE-Math-DeepSeek-7B)]

## 💥 News
- **[2025.04.12]** 🎉 MATHGLANCE benchmark and evaluation code is officially released. 🚀
- **[2024.03.26]** 🎉 The MATHGLANCE paper has been officially uploaded to arXiv [MATHGLANCE](https://arxiv.org/pdf/2503.20745).

## 🔍 Todo
- [x] Update the README.
- [x] Release the benchmark and evaluation code. 
- [ ] Support by lmms-eval [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for very fast evalution 🚀.

## 👀 About MATHGLANCE
The MathGLance benchmark is a novel evaluation framework designed to assess the mathematical perception abilities of Multimodal Large Language Models (MLLMs). Unlike existing benchmarks that often conflate perception with high-level reasoning tasks, MathGLance isolates perceptual skills by focusing on mathematical visual reasoning with minimal cognitive load. It provides both quantitative and qualitative assessments across different granularity levels. For more details, please refer to the project page.

```
benchmark
|-- data
    |-- annotation
        |-- FINAL_COMBINE_MIX_V2.0.0.json
        |-- plane_geometry
            |-- obj_num_1
                |-- images
                |-- all_examples.json 
            |-- ...
        |-- solid_geometry
            |-- ...
        |-- graphs
...
```


## 💪 Evaluation by yourself
Currently, we provide the benchmark data (including images and JSON files) as well as the evaluation code to derive the result scores.

There are two steps for the evaluation of this benchmark, namely inference step and evalution step. Here is an example:

#### Step 1: Inference step

```bash
cd benchmark

python benchmark/scripts/batch_infer_deepseek.sh
```
And the inference results will be saved in `benchmark/outputs/` like `benchmark/outputs/deepseek2vl-small_result.json`

#### Step2: Evaluation step

```bash
cd benchmark

python benchmark/scripts/eval.sh
```
And the statistics of the evaluation results will be saved in `benchmark/outputs/` like `benchmark/outputs/deepseek2vl-small_result.log`
We have pre-configured a set of testing methods, including ["ours", "qwen", "qwen2_5", "deepseek", "llava", "internvl", "internvl2", "internvl_X_2_5", "gpt4o", "gpto1"]. If you have a custom model, you can refer to these methods and add additional ones as needed.

## 🧠 Related Work
Explore our additional research on our **Vision-Language Large Models**, focusing on multi-modal LLMs and mathematical reasoning:
- **[xxx]** [xxx](https://github.com/lupantech/MathVista)