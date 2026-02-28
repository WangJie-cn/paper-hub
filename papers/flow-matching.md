# Flow Matching for Generative Modeling

**arxiv:** [2210.02747](https://arxiv.org/abs/2210.02747) | **作者:** Yaron Lipman 等 (Meta AI + Weizmann Institute) | **年份:** 2022

## 一句话总结
提出 Flow Matching：一种**无需模拟**的训练方法，让 CNF（连续归一化流）可以像 Diffusion 一样大规模训练，且支持更高效的 OT（最优传输）路径。

---

## 核心贡献

1. **Flow Matching 目标函数** — 直接回归速度场 $v_t(x)$，无需跑 ODE 模拟
2. **Conditional Flow Matching (CFM)** — 类似 denoising score matching，每个样本独立训练
3. **OT 路径** — 用最优传输定义直线轨迹，比 diffusion 的弯曲路径更高效

---

## Diffusion vs Flow Matching 对比

| | Diffusion | Flow Matching |
|---|---|---|
| **训练目标** | 预测噪声 ε 或 score | 预测速度场 v |
| **轨迹形状** | 弯曲（可能绕路） | 直线（OT 路径） |
| **数学框架** | SDE（随机微分方程） | ODE（常微分方程） |
| **采样效率** | 需要很多步 | 更少步数 |

---

## 核心公式

### Flow Matching Loss
```
L_FM(θ) = E_{t,p_t(x)} ||v_t(x) - u_t(x)||²
```

### Conditional Flow Matching Loss（实际使用）
```
L_CFM(θ) = E_{t,q(x₁),p(x₀)} ||v_t(ψ_t(x₀)) - (x₁ - (1-σ)x₀)||²
```

### OT 路径（直线插值）
```
ψ_t(x) = (1 - (1-σ_min)t)x₀ + t·x₁
```

---

## 直观理解

- **Diffusion**：加噪 → 学去噪，轨迹是曲线，可能"绕路"
- **Flow Matching**：学习"从噪声漂到数据"的速度场
- **OT 路径**：让轨迹变成直线，最短路径

想象一群粒子从噪声分布出发，Flow Matching 告诉每个粒子"往哪走、走多快"，最终到达数据分布。

---

## 关键结果

- ImageNet 128×128：FID 和 likelihood 都优于 diffusion 方法
- 采样速度：4-10 步出好图（diffusion 通常需要 20-50 步）
- 训练收敛：OT 路径比 diffusion 路径更快

---

## 代码

- **官方实现：** https://github.com/facebookresearch/flow_matching
- **框架：** PyTorch

---

## 视频教程

1. **Yaron Lipman ICML 2023 Tutorial（作者本人）**
   - https://www.youtube.com/watch?v=5ZSwYogAxYg
   
2. **Yannic Kilcher 讲解**
   - https://www.youtube.com/watch?v=DDq_pIfHqLs

---

## 推荐阅读顺序

1. **Section 1-2**：背景 + CNF 基础
2. **Section 3**：Flow Matching 核心（重点看 3.2 CFM）
3. **Section 4.1 Example II**：OT 路径（精华！）
4. **Figure 2-4**：直观对比 diffusion vs OT

---

## 下载

📄 [flow_matching_2210.02747.pdf](../flow_matching_2210.02747.pdf)
