# VQA 训练与 CoT 能力内化机制

> **状态**: 🔬 待深入研究  
> **核心问题**: 为什么用 QA 训练的模型，推理时可以不需要 Question？能力是如何"内化"的？

---

## 1. 核心问题

训练时：
```
Image + "What objects are important?" → "Pedestrian at front"
Image + "What action to take?" → "Slow down"
```

推理时：
```
Image → Trajectory  (不需要 Question，直接输出)
```

**为什么可以去掉 Question？数学上如何解释？**

---

## 2. 直观理解

### 2.1 类比：人类学习

| 阶段 | 学开车 | VLM 训练 |
|-----|-------|---------|
| 学习时 | 教练问"前面有什么？该怎么办？" | QA prompt 引导推理 |
| 熟练后 | 直接反应，不需要自问自答 | 直接 image → action |

### 2.2 QA 的作用

QA 不是目的，而是**训练工具**：
- 强迫模型学会正确的**注意力分配**
- 强迫模型建立**视觉特征 ↔ 语义理解 ↔ 动作决策**的映射

---

## 3. 数学原理 (待研究)

### 3.1 条件概率视角

训练时学习的是：
$$P(\text{answer} | \text{image}, \text{question})$$

但实际上，question 是固定的（比如总是问"该怎么做"），所以：
$$P(\text{action} | \text{image}, q_{\text{fixed}}) \approx P(\text{action} | \text{image})$$

**假设**: 当 question 分布固定时，模型学到的是 image → answer 的直接映射，question 变成了"格式提示"而非"信息输入"。

### 3.2 信息论视角

$$I(\text{Answer}; \text{Question} | \text{Image}) \stackrel{?}{=} 0$$

如果给定 Image，Question 不提供额外信息（因为 Question 是固定模板），那么模型本质上学的是：
$$I(\text{Answer}; \text{Image})$$

### 3.3 特征空间视角

训练前（预训练 VLM）：
```
图像特征空间: 通用的视觉-语言对齐
f(image) ∈ R^d (通用)
```

训练后（QA 微调）：
```
图像特征空间: 被"雕刻"成任务相关的结构
f(image) ∈ R^d (任务特化)

行人特征区域 → 自动激活"减速"相关的特征方向
红灯特征区域 → 自动激活"停车"相关的特征方向
```

**待研究**: 这种"雕刻"的数学描述是什么？是否可以用 representation learning 理论解释？

### 3.4 CoT 压缩假设

完整 CoT：
$$\text{Image} \rightarrow h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow \text{Action}$$
$$h_1 = \text{"行人在前方"}, h_2 = \text{"可能横穿"}, h_3 = \text{"应该减速"}$$

训练后内化：
$$\text{Image} \rightarrow h_{\text{compressed}} \rightarrow \text{Action}$$

**问题**: $h_{\text{compressed}}$ 是否包含了 $h_1, h_2, h_3$ 的信息？如何验证？

---

## 4. 三种实现方案

### 4.1 方案 A: 多任务联合训练

```python
class MultiTaskVLM(nn.Module):
    def __init__(self):
        self.encoder = VisionEncoder()  # 共享
        self.llm = LLM()                # 共享
        
        # 多个输出头
        self.perception_head = Linear(d, vocab)
        self.prediction_head = Linear(d, vocab)
        self.planning_head = Linear(d, vocab)
        self.trajectory_head = Linear(d, num_waypoints * 2)
    
    def forward(self, image, task):
        h = self.llm(self.encoder(image))
        if task == 'trajectory':
            return self.trajectory_head(h)
        else:
            return getattr(self, f'{task}_head')(h)

# Loss
loss = λ1 * L_perception + λ2 * L_prediction + λ3 * L_planning + λ4 * L_trajectory
```

**推理时**: 只用 trajectory_head，其他能力内化在共享的 encoder + llm 中。

### 4.2 方案 B: 两阶段训练

**阶段 1**: 完整 CoT 训练
```python
# 输入: image + CoT_prompt
# 输出: 完整推理链
model.train(image, "Describe and plan") → "<scene>...</scene><action>...</action>"
```

**阶段 2**: 精简微调
```python
# 冻结大部分参数
for p in model.parameters():
    p.requires_grad = False
for p in model.last_layers.parameters():
    p.requires_grad = True

# 只输出轨迹
model.finetune(image, "") → trajectory
```

### 4.3 方案 C: 知识蒸馏

```python
# Teacher: 大模型，完整 CoT
# Student: 小模型，直接输出

teacher_traj = teacher(image, cot_prompt)  # 慢，但准
student_traj = student(image, "")          # 快

loss = MSE(student_traj, teacher_traj) + MSE(student_hidden, teacher_hidden)
```

---

## 5. 车端 HMI 显示 CoT

如果需要在车端**同时**输出轨迹和 CoT 解释：

### 5.1 双输出头 (推荐)

```python
class DualOutputVLM(nn.Module):
    def forward(self, image):
        h = self.llm(self.encoder(image))
        
        # 并行输出
        trajectory = self.traj_head(h[:, 0])      # 快，用于控制
        cot_logits = self.cot_head(h)             # 用于 HMI
        
        return trajectory, cot_logits
```

### 5.2 结构化 CoT (最快)

```python
class StructuredCoTVLM(nn.Module):
    def forward(self, image):
        h = self.llm(self.encoder(image))[:, 0]
        
        return {
            'scene': self.scene_cls(h),      # 分类: 城市/高速
            'weather': self.weather_cls(h),  # 分类: 晴/雨
            'objects': self.obj_det(h),      # 检测: bbox + class
            'action': self.action_cls(h),    # 分类: 17 类 meta-action
            'risk': self.risk_reg(h),        # 回归: 风险分数
            'trajectory': self.traj_reg(h)   # 回归: 轨迹点
        }
```

**优点**: 一次 forward (~50ms) 同时输出所有内容，无需自回归。

### 5.3 异步更新

```python
# 高频线程 (10Hz): 只输出轨迹
# 低频线程 (2Hz): 输出完整 CoT 用于 HMI
```

---

## 6. 待深入研究的问题

### 6.1 理论问题

- [ ] **信息保留**: CoT 压缩后，中间推理步骤的信息是否被保留？如何度量？
- [ ] **泛化性**: 去掉 question 后，模型在 OOD 场景的泛化能力是否下降？
- [ ] **可解释性**: 内化后的模型，如何提取/恢复中间推理过程？

### 6.2 实验验证

- [ ] 对比实验: 有 QA 训练 vs 无 QA 训练，最终轨迹精度差异
- [ ] 消融实验: 不同 CoT 步骤的重要性
- [ ] 可视化: Attention map 在训练前后的变化

### 6.3 相关论文

- [ ] Chain-of-Thought Prompting (Wei et al., 2022)
- [ ] Let's Think Step by Step (Kojima et al., 2022)
- [ ] Distilling Step-by-Step (Hsieh et al., 2023)
- [ ] The Unreasonable Effectiveness of Easy Training Data (Saunders et al., 2022)

---

## 7. 参考链接

- [DriveLM 数据集](/datasets/drivelm-viewer.html)
- [DriveVLM 论文](drivevlm_2402.12289.pdf)
- [DriveVLM CoT 训练详解](/data/drivelm/README_CoT_Training.md)

---

*Last updated: 2026-02-28*
