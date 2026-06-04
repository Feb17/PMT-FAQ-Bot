# Handoff: PMT FAQ Bot 双机 RAG 部署

**生成时间:** 2026-05-13  
**工作区:** `C:\Users\agu2szh\OneDrive - Bosch Group\Documents\H20-GPU`

---

## 目标

将现有 RAG 栈拆成双机部署，**不改动**仓库内既有目录结构（尤其保持 `models/llm-services/docker-compose.yml` 原样，因 xe9680 本机已在跑 Open WebUI）。

| 主机 | IP | 角色 |
|------|-----|------|
| xe9680 | `10.203.97.4` | 基础设施：Qdrant、vLLM Chat、TEI Embedding、TEI Rerank |
| VM-3-17-ubuntu | `10.20.3.17` | 应用：Open WebUI、RAG API（同一 Docker network） |

---

## 已确认决策

1. **TEI Embedding / Rerank** 与 vLLM、Qdrant 一样留在 xe9680。
2. **Open WebUI 与 `rag-api` 在同一 Docker network**；Open WebUI 访问 RAG 使用容器名 `http://rag-api:8088/v1`，不是宿主机 IP。
3. **VM → xe9680** 用内网 IP 访问基础设施（非 Docker 服务名）。
4. **不修改** `models/llm-services/docker-compose.yml`；xe9680 上现有 vLLM/TEI 端口映射已是 `host:container` 形式，默认监听 `0.0.0.0`，本机外可访问（防火墙允许时）。
5. 用户先要**部署说明/配置包**，后在根目录新建 **`PMT-FAQ-Bot/`** 承载新需求文件；未改 `models/`、`confluence-dify-clean-export/` 等既有路径。

---

## 网络与连通性（已实测）

在 **VM**（`root@VM-3-17-ubuntu:/home/ubuntu/pmt-faq-app`）上：

| 端点 | 结果 | 含义 |
|------|------|------|
| `curl http://10.203.97.4:8000/health` | `200 OK` | vLLM 从 VM 可达 |
| `curl http://10.203.97.4:6333/health` | `401 Unauthorized` | Qdrant **端口已通**，需 `api-key` header，不是网络阻断 |

**待 VM 上补测：**

```bash
curl -i http://10.203.97.4:8001/health   # TEI Embedding
curl -i http://10.203.97.4:8002/health   # TEI Rerank
curl -i -H "api-key: <QDRANT_API_KEY>" http://10.203.97.4:6333/readyz
```

Qdrant 绑定：现有 `models/qdrant/.env.example` 默认 `QDRANT_BIND_HOST=127.0.0.1`；若 VM 能拿到 401，说明当前 xe9680 上 Qdrant 已对外监听或经其他方式暴露，与 example 不一致——以 **`ss -ltn`** 与实机 `.env` 为准。

---

## 交付物（新目录，仅此处）

路径：`PMT-FAQ-Bot/`

| 文件 | 用途 |
|------|------|
| `README.md` | 双机拓扑、启动顺序、验证步骤 |
| `compose.app-vm-3-17.yml` | VM：`open-webui`、`rag-api`、可选 `rag-ingest`（profile `ingest`） |
| `app-vm-3-17.env.example` | VM 环境变量模板（含 `OPENAI_API_BASE_URLS`） |
| `open-xe9680-ports.sh` | xe9680 防火墙：仅允许 `10.20.3.17` 访问 6333/6334/8000-8002 |
| `check-vm-to-xe9680.sh` | VM 侧连通性检查 |
| `compose.infra-xe9680.yml` | **可选**；用户选择复用现有 `llm-services` 时**不必使用** |
| `infra-xe9680.env.example` | 配合可选 infra compose |

**VM 侧关键 env（`app-vm-3-17.env.example`）：**

```env
QDRANT_URL=http://10.203.97.4:6333
EMBEDDING_URL=http://10.203.97.4:8001
RERANK_URL=http://10.203.97.4:8002
LLM_URL=http://10.203.97.4:8000
OPENAI_API_BASE_URLS=http://10.203.97.4:8000/v1;http://rag-api:8088/v1
```

`QDRANT_API_KEY` 需与 xe9680 上 Qdrant 一致（参考 `models/qdrant/.env.example`，勿把真实 key 写入仓库）。

**默认 collection 名：** 新包用 `pmt_faq_knowledge`；若复用已有数据，改为现有名（原 RAG 使用 `isacn_knowledge`，见 `models/rag-ingest/.env`）。

**镜像依赖：** `rag-api` / `rag-ingest` 使用 `rag-ingest:latest`，构建来源为 `models/rag-ingest/`（`Dockerfile` + `src/`）。VM 上需已有该镜像或从 xe9680/制品库导入。

---

## 既有代码参考（勿重复展开，按需打开）

| 路径 | 说明 |
|------|------|
| `models/llm-services/docker-compose.yml` | vLLM、TEI、本机 Open WebUI；**保持不动** |
| `models/llm-services/.env` | 模型名、端口 8000/8001/8002 |
| `models/qdrant/compose.yml` + `.env` | Qdrant；注意 `QDRANT_BIND_HOST` |
| `models/rag-ingest/compose.yml` + `src/` | RAG API 实现、健康检查 `/healthz` |
| `docs/architecture/rag-query-flow.*` | 查询链路架构图 |

原单机假设：所有服务共享 external network `llm-net`。双机后 VM 应用栈使用 `pmt-faq-app-net`，仅通过 IP 访问 xe9680。

---

## 未完成 / 下一步

1. **VM：** 在 `pmt-faq-app`（或同步后的 `PMT-FAQ-Bot`）执行 `cp app-vm-3-17.env.example .env`，填入真实 `QDRANT_API_KEY`，确认 `COLLECTION_NAME`。
2. **VM：** 补测 8001/8002；`bash check-vm-to-xe9680.sh`。
3. **VM：** `docker compose -f compose.app-vm-3-17.yml up -d open-webui rag-api`（需 `rag-ingest:latest`）。
4. **xe9680：** 若防火墙拦截，运行 `bash open-xe9680-ports.sh`（**不**要求改 `llm-services` compose）。
5. **可选：** 文档导入 `docker compose --profile ingest run --rm rag-ingest`，挂载 FAQ 文档目录到 `DOCUMENTS_DIR`。
6. **验证：** `curl http://127.0.0.1:8090/healthz`（RAG 调试端口）、Open WebUI `http://10.20.3.17:3000`，确认两个模型源（vLLM + `pmt-faq-bot`）。

**未做：** 在目标机上实际 `docker compose up`；本机 Windows 无 `docker` CLI，未做 compose config 校验。

---

## 约束与坑

- 防火墙放行 ≠ 服务监听外网；若仅 `127.0.0.1`，需改 Qdrant bind（用户当前 6333 已从 VM 收到 401，说明已可达）。
- Qdrant `/health` 无 key → 401；应用与探测应使用 `/readyz` + `api-key` 或 RAG `/healthz`。
- xe9680 本机 Open WebUI（`llm-services` 内）与 VM 上 Open WebUI 是**两套**；不要为 VM 改 xe9680 的 compose。
- 工作区**不是 git 仓库**（曾 `git status` 失败）。

---

## 建议下一会话使用的 Skills

| Skill | 何时用 |
|-------|--------|
| `verification-before-completion` | 在 VM/xe9680 上跑完 compose 与健康检查后，再宣称部署完成 |
| `diagnose` / `systematic-debugging` | 若 RAG `/healthz` 503、Open WebUI 连不上 `rag-api` 或跨机超时 |
| `bosch-procedure-documentation` | 若需把部署步骤写入 Confluence/PIX |
| `writing-plans` | 若要将剩余步骤拆成可执行实施清单（用户此前只要 guide，未写 spec 到 `docs/superpowers/`） |

**一般不需要：** `brainstorming`（方案已定型）、`split-to-prs`（无 git PR 流）。

---

## 用户环境片段

- VM 工作目录已出现：`/home/ubuntu/pmt-faq-app`（可能与仓库中 `PMT-FAQ-Bot` 为拷贝/重命名关系，下一会话先 `ls` 对齐文件名）。
- 用户规则：默认**中文**回复；**未经要求不要 commit**；不要改未请求的现有文件。
