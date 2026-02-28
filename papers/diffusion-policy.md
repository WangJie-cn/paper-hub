# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**arxiv:** [2303.04137](https://arxiv.org/abs/2303.04137) | **作者:** Cheng Chi, Siyuan Feng 等 (Columbia + Toyota Research) | **年份:** 2023

## 一句话总结
用 Diffusion Model 生成机器人动作序列，解决多模态动作分布问题，成为机器人学习的重要 baseline。

---

## 核心贡献

1. **Diffusion 做动作生成** — 首次将 diffusion 用于 visuomotor policy
2. **Action Chunking** — 预测连续多步动作，不是单步
3. **多模态动作分布** — 自然处理"一个状态多种合理动作"的问题

---

## 为什么 Diffusion 适合动作生成？

传统方法（如 BC）用回归预测单一动作，但现实中：
- 同一状态可能有多种合理动作（比如绕左边还是右边）
- 单一预测会"平均"这些动作 → 失败

Diffusion 生成的是**分布**，可以采样不同的合理动作。

---

## 架构

```
观测 (图像 + 状态)
        │
        ▼
┌───────────────────┐
│  Visual Encoder   │
│  (ResNet/ViT)     │
└───────────┬───────┘
            │
            ▼
┌───────────────────┐
│  Diffusion Model  │
│  (U-Net / Trans)  │
│                   │
│  输入: 噪声动作    │
│  条件: 观测特征    │
│  输出: 去噪动作    │
└───────────┬───────┘
            │
            ▼
    Action Chunk
    [a_t, a_{t+1}, ..., a_{t+H}]
```

---

## 关键设计

### 1. Action Chunking
- 预测未来 H 步动作（如 16 步）
- 执行时用滑动窗口
- 好处：时间一致性、平滑

### 2. DDPM vs DDIM
- 训练用 DDPM（标准 diffusion）
- 推理用 DDIM（加速采样）
- 通常 10-20 步采样

### 3. 观测历史
- 输入最近 T 帧观测
- 捕捉动态信息

---

## 实验结果

在 RoboMimic、PushT 等 benchmark 上大幅超越：
- Behavior Cloning
- IBC (Implicit BC)
- BeT (Behavior Transformer)

---

## 局限性

1. **采样慢** — 需要多步去噪
2. **实时性** — 对高频控制有挑战

这些问题被 **Flow Matching** 改进版本解决（如 π0）

---

## 代码

- **官方实现：** https://github.com/real-stanford/diffusion_policy
- **项目主页：** https://diffusion-policy.cs.columbia.edu/

---

## 后续工作

- **3D Diffusion Policy (DP3)** — 加入 3D 点云
- **π0** — 改用 Flow Matching
- **Octo** — 开源机器人基础模型

---

## 下载

📄 需要时可下载: https://arxiv.org/pdf/2303.04137.pdf
