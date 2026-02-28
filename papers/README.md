# 📚 论文学习中心

## 🎯 学习路径：Flow Matching → VLA

```
                    ┌─────────────────────────────────┐
                    │     1. Flow Matching 基础        │
                    │     (2210.02747)                │
                    │     理解核心数学框架              │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │     2. Diffusion Policy          │
                    │     (2303.04137)                │
                    │     理解动作生成的 baseline       │
                    └───────────────┬─────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          │                                                   │
┌─────────▼─────────┐                           ┌─────────────▼─────────────┐
│  3a. 3D Diffusion │                           │  3b. π0                    │
│  Policy (DP3)     │                           │  (2410.24164)             │
│  3D 点云输入       │                           │  VLM + Flow Matching      │
└───────────────────┘                           └─────────────┬─────────────┘
                                                              │
                                                ┌─────────────▼─────────────┐
                                                │  4. 自动驾驶 VLA 应用       │
                                                │  - DriveVLM               │
                                                │  - GenAD                  │
                                                │  - 你自己的工作！          │
                                                └───────────────────────────┘
```

---

## 📄 已整理论文

| 论文 | 主题 | 状态 | PDF |
|------|------|------|-----|
| [Flow Matching](flow-matching.md) | 基础方法 | ✅ 完成 | [下载](../flow_matching_2210.02747.pdf) |
| [π0](pi0.md) | VLM + Flow | ✅ 完成 | [下载](../pi0_2410.24164.pdf) |
| [Diffusion Policy](diffusion-policy.md) | 动作生成 | ✅ 完成 | [arxiv](https://arxiv.org/pdf/2303.04137.pdf) |

---

## 🔜 待添加

- [ ] 3D Diffusion Policy (DP3)
- [ ] Rectified Flow
- [ ] Conditional Flow Matching
- [ ] DriveVLM
- [ ] GenAD

---

## 🎬 视频资源

### Flow Matching
- [Yaron Lipman ICML Tutorial](https://www.youtube.com/watch?v=5ZSwYogAxYg) — 作者本人，强烈推荐
- [Yannic Kilcher 讲解](https://www.youtube.com/watch?v=DDq_pIfHqLs)

### π0
- [官方演示视频](https://physicalintelligence.company/blog/pi0)

---

## 💡 核心概念速查

### Diffusion vs Flow Matching

| | Diffusion | Flow Matching |
|---|---|---|
| 训练目标 | 预测噪声 ε | 预测速度场 v |
| 轨迹 | 弯曲 | 直线 (OT) |
| 数学 | SDE | ODE |
| 采样步数 | 20-50 | 4-10 |

### 为什么 Flow Matching 适合机器人/自动驾驶？

1. **实时性** — 少步采样，延迟低
2. **平滑性** — 直线轨迹，动作不抖
3. **稳定性** — 训练更简单

---

## 📁 文件结构

```
papers/
├── README.md           # 本文件
├── flow-matching.md    # Flow Matching 论文笔记
├── pi0.md              # π0 论文笔记
├── diffusion-policy.md # Diffusion Policy 论文笔记
└── ...                 # 更多论文笔记
```

---

*最后更新: 2026-02-27*
