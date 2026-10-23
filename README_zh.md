<div align="center">
  <h1>🧠 MemCoRe（🎉 NeurIPS 2026）</h1>
  <p><strong>Recovering Evidence from Progressively Compressed Factual Knowledge<br>for Agent Memory</strong></p>
  <p>
    <a href="https://arxiv.org/abs/2602.07885"><img src="https://img.shields.io/badge/arXiv-2602.07885-b31b1b" alt="论文：arXiv 2602.07885"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-208b74" alt="许可证：MIT"></a>
    <a href="AGENTS.md"><img src="https://img.shields.io/badge/Agent-Guide-208b74" alt="Agent 使用指南"></a>
    <a href="CITATION.bib"><img src="https://img.shields.io/badge/Citation-BibTeX-208b74" alt="BibTeX 引用"></a>
    <a href="README.md"><img src="https://img.shields.io/badge/README-English-208b74" alt="English README"></a>
    <a href="https://github.com/Liar-zzy/MemCoRe/stargazers"><img src="https://img.shields.io/github/stars/Liar-zzy/MemCoRe?style=social" alt="GitHub Stars"></a>
  </p>
</div>

MemCoRe 将智能体记忆组织成 **Note、Keyword 和 Topic** 的层次结构，逐步压缩冗余，同时保留检索证据所需的结构。方法结合门控记忆构建、多层检索与迭代证据细化，为后续问答恢复相关信息。

本次 release 提供 MemCoRe 实现和 LoCoMo 评测流程，默认使用 **Qwen3-8B** 与 **Qwen3-Embedding-0.6B**。

![MemCoRe 方法总览：记忆构建、压缩层次与记忆检索，取自论文 Figure 2](assets/memcore-overview.png)

<p align="center"><em>MemCoRe 方法总览：记忆构建、渐进压缩与证据检索。</em></p>

## 🚀 快速开始

### 1. 📦 安装依赖

使用已验证的 Python 3.11 环境：

```bash
git clone https://github.com/Liar-zzy/MemCoRe.git
cd MemCoRe
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

直接用 `python run.py` 运行，无需安装 MemCoRe 包。评测程序运行在 CPU 上，通过 API 调用聊天模型和 embedding 模型。

### 2. 🔌 配置模型服务

启动模型服务后设置地址，以下也是默认值：

```bash
export MEMCORE_LLM_BASE_URL=http://127.0.0.1:8000/v1
export MEMCORE_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
```

如需认证，设置 `MEMCORE_LLM_API_KEY` 和 `MEMCORE_EMBEDDING_API_KEY`。模型名称与生成参数在 [configs/llm.yaml](configs/llm.yaml) 中；默认关闭 Qwen3 的 thinking。

<details>
<summary>🖥️ 使用 vLLM 启动本地模型（两张 GPU）</summary>

为模型服务创建独立环境：

```bash
conda create -n memcore-serve python=3.11 -y
conda activate memcore-serve
python -m pip install vllm==0.10.0
```

启动聊天模型：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --seed 42 --max-model-len 32768 --max-num-seqs 32 \
  --gpu-memory-utilization 0.80 --enable-prefix-caching \
  --generation-config vllm \
  --override-generation-config '{"max_new_tokens":8192}'
```

在另一终端激活模型服务环境，启动 embedding 模型：

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Embedding-0.6B \
  --served-model-name Qwen/Qwen3-Embedding-0.6B \
  --host 127.0.0.1 --port 8001 \
  --task embed --dtype bfloat16 --tensor-parallel-size 1 \
  --seed 42 --max-model-len 32768 \
  --gpu-memory-utilization 0.20
```

以上命令对应复现使用的两张 H800。其他硬件可调整显存占比；使用本地权重时，替换 `--model` 路径并保留服务模型名称。远程服务可通过可访问的地址或 SSH 端口转发连接。

客户端使用请求 seed 42、top-p 1.0，C1–C4 回答温度为 0.7，C5 为 0.5。Embedding 输出为 1,024 维，建图和检索使用同一个模型。

执行下方评测命令前，切回评测环境与仓库目录。

</details>

<details>
<summary>🔄 切换到 Qwen3-14B 或 GPT-4o</summary>

使用 Qwen3-14B 时，将上方聊天服务命令中的两个模型名称都替换为 `Qwen/Qwen3-14B`，然后运行：

```bash
python run.py --llm-config-path configs/llm_qwen3_14b.yaml --name locomo_qwen3_14b
```

使用 GPT-4o 时，在环境中设置 `OPENAI_API_KEY`，并使用独立配置：

```bash
python run.py --llm-config-path configs/llm_gpt4o.yaml --name locomo_gpt4o
```

此配置调用 OpenAI Chat Completions API，不携带 Qwen/vLLM 专用的请求参数。如需使用其他供应商的地址或可用的模型快照，添加 `--base-url` 或 `--model`。OpenAI 已宣布其 GPT-4o 快照于 2027 年 1 月 14 日下线，具体见[官方下线计划](https://developers.openai.com/api/docs/deprecations)。

两份配置都继续使用独立的 **Qwen3-Embedding-0.6B** 服务。聊天模型的切换作用于记忆构建、查询分析和回答；不同 `--name` 分别保存图和结果。如需先运行小样例，添加 `--dataset examples/smoke_locomo.json` 并使用单独的运行名称。

</details>

### 3. 🧪 先运行小样例

仓库附带六条对话、五个问题的合成样例，覆盖 LoCoMo 的五类问答：

```bash
python run.py --dataset examples/smoke_locomo.json --name demo
python scripts/summarize.py results/demo
```

这个命令会完成建图、检索、回答和评分。结果位于 `results/demo/`，图缓存位于 `cached/cached_mem_key_graph_v3/demo/`。

### 4. 📊 运行完整 LoCoMo

```bash
python scripts/download_data.py
python run.py
python scripts/summarize.py results/locomo_qwen3_8b
```

下载脚本从 [LoCoMo 官方仓库](https://github.com/snap-research/locomo) 获取数据。默认评测全部 **10 段对话、1,986 个问题**，依次为每段对话建图并回答其问题。

只运行第一段对话：

```bash
python run.py --sample-indices 0 --name locomo_sample0
```

中断后重新执行相同命令，会复用图缓存并接续已完成的问答。修改模型或实验参数时，换一个 `--name`。

<details>
<summary>🗂️ 图存储：为什么本次 release 不需要 Neo4j？</summary>

原始实现使用 Neo4j。Community Edition 的一个 DBMS 实例只支持一个用户数据库，因此，多个需要独立图状态的实验需要额外处理应用层隔离，或部署多个实例。具体限制见 [Neo4j 官方文档](https://neo4j.com/docs/operations-manual/current/database-administration/)。

为了简化部署、方便并行实验并支持后续算法扩展，本次 release 整理了轻量的进程内图实现：使用 NumPy 完成向量检索，使用 NetworkX/graspologic 完成图分析与 Topic 聚类。每段对话持有独立的图，可以通过 JSON 保存和恢复，便于管理图状态的隔离、重置与快照，无需额外部署数据库服务。

</details>

## 📖 引用

如果使用本工作，请引用：

```bibtex
@misc{zhang2026memcorerecoveringevidenceprogressively,
      title={MemCoRe: Recovering Evidence from Progressively Compressed Factual Knowledge for Agent Memory},
      author={Zhenyuan Zhang and Xianzhang Jia and Zhiqin Yang and Zhenbo Song and Wei Xue and Sirui Han and Yike Guo},
      year={2026},
      eprint={2602.07885},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.07885},
}
```

## 🙏 致谢

实现基于 [A-MEM](https://github.com/WujiangXu/A-mem) 的代码发展而来，感谢 A-MEM 和 [LoCoMo](https://github.com/snap-research/locomo) 作者提供的开源实现与数据。

## 📄 许可证

代码采用 [MIT License](LICENSE)，保留原有版权声明。数据集与模型权重从各自项目获取。
