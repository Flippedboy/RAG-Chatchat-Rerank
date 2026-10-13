# Rerank 重排改造说明

> 本文档说明本仓库相对于上游 LangChain-Chatchat 的二次开发内容：在知识库检索链路中新增 **BGE-Reranker（CrossEncoder）精排重排模块**，并预留可开关配置。

## 一、改造目标

原生 Chatchat 仅在向量库中做「向量相似度粗排」，容易召回语义不相关、关键词相似的噪声片段，进而诱发大模型幻觉。本次改造在 **向量粗排召回之后、Prompt 组装之前** 插入一道 **CrossEncoder 精排重排**，实现工业界标准的「粗排海量筛选 + 精排精准提纯」双阶段检索。

```
原生：Embedding 向量召回 → Prompt → LLM
改造：Embedding 向量召回 →【Rerank 精排过滤】→ Prompt → LLM
```

## 二、改动文件清单

| 文件 | 改动类型 | 说明 |
| --- | --- | --- |
| `libs/chatchat-server/chatchat/settings.py` | 修改 | 在 `KBSettings` 中新增 Rerank 相关配置项，支持开关与参数 |
| `libs/chatchat-server/chatchat/server/knowledge_base/kb_service/base.py` | 修改 | 新增 `get_reranker()` 懒加载单例；在 `KBService.search_docs()` 返回前注入重排逻辑 |
| `libs/chatchat-server/pyproject.toml` | 修改 | 显式声明 `sentence-transformers` 依赖（CrossEncoder 运行所需） |

> 注入点选择 `KBService.search_docs()` 的原因：它是 FAISS / Milvus / PG / Chroma / ES 等所有向量库子类共用的检索入口，**一处改动全局生效**，无需逐个修改各向量库实现类。

## 三、实现要点

1. **复用项目内置重排器**：直接使用 `chatchat.server.reranker.reranker.LangchainReranker`（基于 `sentence_transformers.CrossEncoder`），其 `compress_documents(documents, query)` 会对「Query + 每个片段」联合打分并按相关性返回 `top_n` 个文档，同时在文档 `metadata` 中写入 `relevance_score`。
2. **模型懒加载单例**：`get_reranker()` 全局缓存重排模型，仅在首次开启重排时加载一次，避免每次问答重复加载模型导致延迟。
3. **开关控制**：由 `Settings.kb_settings.USE_RERANKER` 控制，默认关闭；关闭时完全不加载模型、不影响原有链路。
4. **失败兜底**：重排过程出错（如模型下载失败、显存不足）会捕获异常并回退到原始向量召回结果，保证服务可用。

## 四、配置说明

配置项位于 `KBSettings`，对应运行时的 `kb_settings.yaml`。首次启动会从默认值生成配置文件，也可手动修改：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `USE_RERANKER` | `false` | 是否启用 Rerank 重排。**默认关闭**，按需改为 `true` 开启 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | 重排模型名称或本地路径，可换 `bge-reranker-large` 等 |
| `RERANKER_TOP_N` | `3` | 重排后保留的文档条数（送入 LLM 的片段数） |
| `RERANKER_MAX_LENGTH` | `512` | 单条输入最大长度，超长截断，需与 chunk 大小配合 |
| `RERANKER_DEVICE` | `cpu` | 运行设备，有 GPU 可改为 `cuda` |
| `RERANKER_BATCH_SIZE` | `32` | CrossEncoder 批处理大小 |

开启方式示例（编辑 `kb_settings.yaml`）：

```yaml
USE_RERANKER: true
RERANKER_MODEL: BAAI/bge-reranker-base
RERANKER_TOP_N: 3
RERANKER_MAX_LENGTH: 512
RERANKER_DEVICE: cpu
```

> 注意：`RERANKER_TOP_N` 应小于等于向量粗排召回数 `VECTOR_SEARCH_TOP_K`，否则重排没有候选可筛。

## 五、效果对比方法（A/B 对照）

在**同一知识库、同一组问题**下做开关对照：

1. **基线组（关闭重排）**：`USE_RERANKER: false`，对 5~10 个代表性问题（含易混淆、跨主题问题）逐一问答，记录：
   - 召回片段内容与数量；
   - LLM 最终回答；
   - 单次问答总耗时（可在日志中观察）。
2. **实验组（开启重排）**：`USE_RERANKER: true`，使用完全相同的问题清单重新问答，记录同样三项数据。
3. **对照观察点**：
   - **召回质量**：重排后返回的片段是否更贴合问题、是否剔除了原本混入的无关片段（文档 `metadata.relevance_score` 可辅助判断）；
   - **回答质量**：实验组是否减少了「答非所问 / 编造细节 / 引用了无关文档」的幻觉现象；
   - **耗时代价**：实验组单次问答增加的耗时（主要来自 CrossEncoder 推理），确认在可接受范围。
4. **记录方式**：每个问题截图保存开关前后的界面对比，放入 `assets/`（`rag_old_result.png` / `rag_new_result.png`），形成可复现的案例集。

> 建议同时观察日志：开启重排后每次检索会打印 `Rerank applied: N docs after rerank ...`，便于确认重排确实生效。

## 六、局限性与后续方向

- 重排为串行 CrossEncoder 推理，会小幅增加问答耗时；
- 受 `RERANKER_MAX_LENGTH`（默认 512）限制，超长 chunk 会被截断；
- 纯向量 + 重排对「关键词精确匹配」帮助有限，后续可叠加 BM25 多路召回，再统一送入 Rerank；
- 可引入 RAGAS 等工具对召回率、忠实度做量化评估，替代人工肉眼对比。
