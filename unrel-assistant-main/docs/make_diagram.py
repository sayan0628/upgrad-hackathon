"""Generate docs/workflow_orchestration.png (300 DPI) as a fixed swimlane diagram.
Usage: python3 docs/make_diagram.py   (needs matplotlib)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

OUT = Path(__file__).resolve().parent / "workflow_orchestration.png"
fig = plt.figure(figsize=(24, 28))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-14, 104.5); ax.axis("off")

BLUE, GREEN, RED, PURP, BROWN, GOLD, GREY = "#1f4e79", "#2e7d32", "#c62828", "#6a3d9a", "#8c564b", "#a67c00", "#666666"

def band(y0, y1, title, color, fill):
    ax.add_patch(FancyBboxPatch((1, y0), 98, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fill, ec=color, lw=2.5, zorder=0))
    ax.text(5.5, y1 - 0.8, title, fontsize=15, fontweight="bold", color=color, va="top", zorder=1)

def box(x, y, w, h, text, fc="white", ec="#333", shape="box", fs=10.5):
    if shape == "oval":
        ax.add_patch(Ellipse((x, y), w, h, fc=fc, ec=ec, lw=1.6, zorder=3))
    else:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                    fc=fc, ec=ec, lw=1.6, zorder=3))
    lines = text.split("\n"); lh = 1.5
    y0 = y + (len(lines) - 1) * lh / 2
    for i, l in enumerate(lines):
        ax.text(x, y0 - i * lh, l, ha="center", va="center", zorder=4,
                fontsize=fs + 0.5 if i == 0 else fs - 0.8, fontweight="bold" if i == 0 else "normal")

def arrow(pts, color=BLUE, lw=2.2, ls="-", label=None, lpos=0, loff=(0, 0.9), both=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2)
    kw = dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=kw, zorder=5)
    if both:
        ax.annotate("", xy=pts[0], xytext=pts[1], arrowprops=kw, zorder=5)
    if label:
        a, b = pts[lpos], pts[lpos + 1]
        ax.text((a[0] + b[0]) / 2 + loff[0], (a[1] + b[1]) / 2 + loff[1], label, fontsize=9.5, color=color,
                ha="center", va="center", zorder=6, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))

def line(pts, color, lw=2.2, ls="-"):
    xs, ys = zip(*pts); ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2)

# ---------- title
ax.text(50, 102.6, "Kestrel → Enterprise-Grade Workflow Orchestration", ha="center", fontsize=27, fontweight="bold")
ax.text(50, 100.0, "MCP (JSON-RPC 2.0)  ·  3-Stage Hybrid RAG  ·  Bounded ReAct  ·  Tier-3 HITL  ·  Hosted mid-tier LLM  ·  50,000 VUs  ·  fast paths p95 ≤ 250 ms · LLM path p95 ≤ 2.2 s  ·  36 KPIs   (design v0.2)",
        ha="center", fontsize=14, color=GREY)

# ============ ZONE 1 INGRESS & SECURITY
band(86.5, 98.3, "① INGRESS & SECURITY LAYER   (KPI-01/02/03/27/28)", BLUE, "#eaf2fb")
Y1 = 91.2
box(8.5, Y1, 12, 6, "Clients / Streamlit\n50K VUs · no login\nX-Tenant-Id · X-Role", shape="oval", fs=10)
box(24, Y1, 15, 6.5, "API Gateway\nEnvoy · TLS · no auth (D-22)\nrate-limit · traceparent")
box(41, Y1, 16, 6.5, "FastAPI Workers\nstateless · async · SSE stream\nMCP host client · Pydantic v2")
box(59, Y1, 17, 6.5, "PRE-HOOK 1: PII Redactor\nprio 10 · fail-closed\nPresidio+regex · recall ≥ 99.8%", fc="#cfe3f6")
box(77, Y1, 17, 6.5, "PRE-HOOK 2: Injection Shield\nprio 20 · fail-closed\nblock ≥ 99.5% · hook.pre ≤ 15 ms", fc="#cfe3f6")
box(92.5, Y1, 11, 8.6, "Cache (gateway-direct)\nexact SHA256(t:u:q) ∥\nembed → tenant NN\nTTL 300±45 s", fc="#fde2e2", fs=9)
for a, b in ((14.5, 16.5), (31.5, 33), (49, 50.5), (67.5, 68.5), (85.5, 87)):
    arrow([(a, Y1), (b, Y1)])

# ============ ZONE 2 TRIAGE
band(67.6, 85.4, "② TRIAGE & SUBAGENT ROUTING   (KPI-04/05/06)", GREEN, "#eaf6ea")
box(92.5, 79.6, 11, 7, "Triage Classifier\ndistilled · ≤ 25 ms\nintent · domain ·\nrisk · complexity", fc="#c9ebc9", fs=9.5)
arrow([(92.5, 86.9), (92.5, 83.1)], color=BLUE, label="MISS", loff=(1.7, 0))
RY, BUS = 71.7, 76.3
box(10, RY, 12, 6, "Reject Out-of-Scope\nREJECT_OOS\ndeterministic refusal", fc="#fff2cc")
box(28, RY, 17, 6, "Execution Subagent\nTRANSACTION_EXEC\nschema validation · no RAG/web")
box(47, RY, 17, 6, "Retrieval Subagent\nread-only · cited · no LLM rewrite\nextractive fast path (no LLM)")
box(66, RY, 14, 6, "Direct Response\nDIRECT_ANSWER\nno retrieval")
line([(92.5, 76.1), (92.5, BUS), (10, BUS)], GREEN)
for x, t in ((10, "REJECT_OOS"), (28, "TRANSACTION_EXEC"), (47, "HYBRID_RAG"), (66, "DIRECT_ANSWER")):
    arrow([(x, BUS), (x, RY + 3)], color=GREEN)
    ax.text(x + 0.7, BUS + 0.9, t, fontsize=9.5, color=GREEN, fontweight="bold", zorder=6)

# ============ ZONE 3 CONTROL PLANE
band(52.2, 66.4, "③ CONTROL PLANE — BOUNDED REACT LOOP   (KPI-14..19)", RED, "#fcecec")
CY = 57.5
box(14, CY, 17, 7.4, "GoalAnchorBlock @ idx 0\nimmutable intent + entity IDs\ncompressor ≤180 tok · 3 turns", fs=10)
box(33, CY, 17, 8.8, "Bounded ReAct Machine\nINIT→TRIAGE→REASONING→\nTOOL_CALL→OBSERVATION→VERIFY\n→DONE|HALTED · max 5 steps\n≤120 ms/tool step · 0.6 s / 3.5 s deadline", fc="#f7c9c9", fs=9.5)
box(52, CY, 18, 7.4, "ToolExecutionHarness (TOOL_CALL)\n1 Pydantic strict · 2 repair (T3: shown)\n3 RBAC caller×tool · 4 idem. (T2/T3)", fs=10)
box(71, CY, 16, 7.4, "Circuit Breaker\n(tenant, tool, args_hash)\nfails ≥ 2 → HALTED_BREAKER", fc="#ffd0d0", fs=10)
box(90, CY, 14, 7.4, "LLM: Anthropic via OpenRouter\nHaiku 5.5 · reasoning off · TTFT ≈ 0.7 s\nfallback Haiku 4.5 · pool 256 (quota)", fs=9.5)
arrow([(22.5, CY), (24.5, CY)]); arrow([(41.5, CY), (43, CY)])
arrow([(61, CY), (63, CY)])
# routes into anchor / llm
line([(28, RY - 3), (28, 67.0), (14, 67.0)], BLUE); line([(47, RY - 3), (47, 67.0), (28, 67.0)], BLUE)
arrow([(14, 67.0), (14, CY + 3.7)], color=BLUE)
line([(66, RY - 3), (66, 67.0), (91, 67.0)], BLUE); arrow([(91, 67.0), (91, CY + 3.7)], color=BLUE)
# reasoning loop
arrow([(33, CY + 4.4), (33, 63.3), (88, 63.3), (88, CY + 3.7)], color="#7f1d1d", both=True, label="REASONING ⇄ draft tokens", lpos=1, loff=(0, 0.0))

# ============ ZONE 4 MCP + RAG
band(27.2, 51.0, "④ MCP SERVERS (JSON-RPC 2.0) & 3-STAGE HYBRID RAG   (KPI-07..13, 26, 30)", PURP, "#f3ecf9")
AY = 42.0
box(90, AY, 14, 8.2, "MCP: mcp-hybrid-rag\nstdio · timeout 110 ms\nrag_pool 64 · tool\nhybrid_retrieve · rag://chunks/{id}", fc="#dccdee", fs=9.5)
box(71, 46.0, 16, 6.2, "Stage 1 · Sparse BM25\nconfigurable k1 1.25 · b 0.75\nper-tenant index · top-50 · < 12 ms", fs=9.5)
box(71, 38.5, 16, 6.2, "Stage 2 · Dense (FAISS HNSW)\nembedding reused from gateway\nefSearch 64 · top-50 · < 8 ms", fs=9.5)
box(52, AY, 12, 7, "RRF Fusion\nk = 60 · dedup hash\npool 40→16–20 · <3 ms", fs=9.5)
box(33, AY, 17, 8.2, "Stage 3 · Cross-Encoder\nbge-reranker-v2-m3 ONNX FP16\nbatch 32 · keep top-4\nscore ≥ 0.42 passes · <45 ms", fc="#dccdee", fs=9.5)
box(10.5, AY, 14, 7, "ONNX / Triton workers\nmicro-batch 32\nmax_wait 8 ms", shape="box", fs=9.5)
arrow([(83, AY + 2.5), (79.2, 46.0)], color=PURP, lw=1.8)
arrow([(83, AY - 2.5), (79.2, 38.5)], color=PURP, lw=1.8)
arrow([(63, 46.0), (58.2, AY + 1.5)], color=PURP, lw=1.8)
arrow([(63, 38.5), (58.2, AY - 1.5)], color=PURP, lw=1.8)
arrow([(46, AY), (41.5, AY)], color=PURP, lw=2.2)
arrow([(25.5 - 1, AY - 2.5), (17.5, AY - 2.5)], color=PURP, ls=":", lw=1.6, both=True)
ax.text(21.0, AY - 4.2, "batched inference", fontsize=8.5, color=PURP, ha="center", va="center", zorder=6)
# ce -> react (observation)
arrow([(33, AY + 4.3), (33, CY - 4.6)], color=PURP, ls="--")
ax.text(34.2, 48.3, "top-4 chunks + confidence → OBSERVATION", fontsize=9.5, color=PURP, fontweight="bold", va="center", zorder=6)
# MCP lower row
SY = 31.8
box(54, SY, 22, 6, "MCP: mcp-state-store\nStreamable HTTP · pool 200 · db_pool 128\nsession_facts · T3 approvals · audit", fc="#dccdee", fs=9.5)
box(79, SY, 18, 7, "MCP: mcp-graphify-vault\nstdio · 1–2 hop [[wikilinks]]\n~320 vs ~1,400 tokens (−45–65%)\npruned subgraph → ReAct context", fc="#dccdee", fs=9.5)
# dispatch from circuit breaker
DB = 52.7
line([(71, CY - 3.7), (71, DB), (90, DB)], PURP, 2.4)
arrow([(90, DB), (90, AY + 4.1)], color=PURP, lw=2.4)
arrow([(81, DB), (81, SY + 3.5)], color=PURP, lw=2.4)
arrow([(62, DB), (62, SY + 3)], color=PURP, lw=2.4)
line([(71, DB), (62, DB)], PURP, 2.4)
ax.text(83.5, DB, "JSON-RPC 2.0 dispatch", fontsize=10, color=PURP, fontweight="bold", ha="center", zorder=6,
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
# graphify pruned subgraph back to react: left margin route along y=34.5? keep as note

# ============ ZONE 5 DATA
band(14.6, 26.4, "⑤ DATA & KNOWLEDGE STORES", BROWN, "#f6eeea")
DY = 19.6
box(43, DY, 22, 8.4, "PostgreSQL + pgvector\nchunks · session_facts · approvals\napproval_payloads (KMS, TTL) · RLS\nper-tenant hash-chained audit log", shape="box", fc="#fde2e2", fs=9.2)
box(65, DY, 17, 8.4, "Redis Cluster\ncache (exact + semantic)\nhot window (last 3 turns)\nplatform-owned · tenant keys", fc="#fde2e2", fs=9.2)
box(86, DY, 20, 8.4, "Obsidian Vault + graph.json\nknowledge_vault/*.md\natomic notes · YAML · wikilinks", fc="#fff8dc", fs=9.5)
box(15.5, DY, 20, 8.4, "Index Store (per tenant)\nFAISS HNSW + BM25 · in RAM\nblue/green · canary · from chunks\nread by mcp-hybrid-rag", fc="#efe4f7", fs=9.2)
arrow([(50, SY - 3), (44, DY + 4.2)], color=BROWN, lw=1.8)
arrow([(79, SY - 3.5), (84, DY + 4.2)], color=BROWN, lw=1.8)

# ============ ZONE 6 EGRESS
band(0.4, 13.2, "⑥ VERIFICATION, HITL GATE & EGRESS   (KPI-19..25)", GOLD, "#fdf6e3")
EY = 6.3
box(13, EY, 18, 7.4, "Deterministic Abstention\ntop score < 0.42 → no LLM\ntemplate answer · < 90 ms", fc="#fff2cc", fs=10)
box(34, EY, 19, 7.4, "Tier 3 HITL Gate (async)\nPENDING_APPROVAL · encrypted payload\nreviewer API · reviewer ≠ requester", fc="#ffe3a3", fs=9.5)
box(55, EY, 17, 7.4, "Streamed Response (SSE)\nevidence → token* → [retract] → final\nfast ≤ 250 ms · LLM p95 ≤ 2.2 s", shape="oval", fc="#d9f2d9", fs=9.5)
box(74, EY, 18, 7.4, "POST-HOOK: RAGAS Auditor\nsampled · async · alert-only · hosted judge\nfaith ≥ 0.92 · rel ≥ 0.90 · ent ≥ 0.85", fs=9.2)
box(92, EY, 13, 7.4, "Verification\nSubagent\ninline NLI (self-hosted)\nveto → retract", fs=9.5)
arrow([(92 - 6.5, EY), (74 + 9, EY)], color=GOLD)
arrow([(74 - 9, EY), (55 + 8.5, EY)], color=GOLD, label="pass", loff=(0, 1.0))
# VERIFY from LLM (right margin)
line([(97, CY), (98.4, CY), (98.4, EY + 6)], GOLD, 2.2); arrow([(98.4, EY + 6), (95.6, EY + 3.7)], color=GOLD)
ax.text(98.9, 30, "VERIFY", fontsize=9.5, color=GOLD, fontweight="bold", rotation=90, ha="center", va="center", zorder=6)
# abstention from cross-encoder (left margin)
line([(28, AY - 4.1), (28, 13.9), (13, 13.9)], GOLD, 2.2)
arrow([(13, 13.9), (13, EY + 3.7)], color=GOLD)
ax.text(29.2, 30.5, "score < 0.42", fontsize=9.5, color=GOLD, fontweight="bold", va="center", zorder=6)
# abstention -> response
line([(13, EY - 3.7), (13, 1.6), (55, 1.6)], GOLD, 2.2); arrow([(55, 1.6), (55, EY - 3.5)], color=GOLD)
# Postgres -> HITL
arrow([(36, DY - 4.2), (34, EY + 3.7)], color=GOLD, lw=2.2, label="PENDING_APPROVAL", loff=(9.0, 0))
# refusal & cache-hit & halted breaker
line([(10, RY - 3), (10, 69), (2.4, 69), (2.4, 0.9), (50, 0.9)], GOLD, 2.0, ls="--"); arrow([(50, 0.9), (50, EY - 3.3)], color=GOLD, ls="--", lw=2.0)
ax.text(2.4, 40, "refusal / blocked", fontsize=9.5, color=GOLD, fontweight="bold", rotation=90, ha="center", va="center", zorder=6, bbox=dict(fc="white", ec="none", pad=1))
line([(99.4, Y1), (99.4, 1.2), (60, 1.2)], GREEN, 2.2); arrow([(60, 1.2), (60, EY - 3.4)], color=GREEN)
ax.text(99.4, 70, "cache HIT → skip pipeline (≈ 25 ms)", fontsize=9.5, color=GREEN, fontweight="bold", rotation=90, ha="center", va="center", zorder=6, bbox=dict(fc="white", ec="none", pad=1))

# ============ ZONE 7 OBSERVABILITY + LEGEND
band(-12.8, -1.4, "⑦ OBSERVABILITY (W3C traceparent on every hop)", GREY, "#f2f2f2")
box(25, -7.4, 38, 6.4, "OTel Collector (otel_exporter · ALL_SPANS)\nhook.pre · cache.lookup · gateway.embed · triage.classify · mcp.rag.{bm25,faiss,rrf,rerank}\nagent.react.step · llm.generate · hook.post.ragas", fs=9.2)
box(57, -7.4, 20, 6.4, "Prometheus + Tempo + Grafana\n36-KPI scorecard · alerts\nincident runbooks", shape="box", fs=9.5)
arrow([(44, -7.4), (47, -7.4)], color=GREY)
ax.text(10, -11.2, "SLA load run: LLMClient → calibrated llm-sim (replays measured model latency/429s); real-model soak validates it", fontsize=9.2, color=GREY, zorder=6)
ax.text(10, -12.2, "async, non-blocking export from all zones ①–⑥", fontsize=9.5, color=GREY, zorder=6)
lx = 74
for i, (c, ls, t) in enumerate(((BLUE, "-", "Main request flow"), (GREEN, "-", "Triage routing / cache hit"), (PURP, "-", "MCP JSON-RPC / RAG"),
                                (GOLD, "-", "Abstention · HITL · verification"), (GOLD, "--", "Refusal / block / halted"))):
    yy = -3.6 - i * 1.7
    ax.plot([lx, lx + 3.5], [yy, yy], color=c, lw=2.5, ls=ls, zorder=3); ax.text(lx + 4.3, yy, t, fontsize=10, va="center", zorder=4)

fig.savefig(OUT, dpi=300)
print("wrote", OUT)
