
# Claude.md: 光纤通信非线性补偿论文项目指南

## 1. 项目背景与目标

* **课题名称：** 基于深度学习的光纤传输系统非线性损伤补偿研究
* **核心任务：** 利用 Claude Code 的 Agent 能力，完成从学术调研、数据仿真（Python/PyTorch）、论文撰写到 LaTeX 排版的完整流程。
* **目标输出：** 一篇结构完整、数据真实、图表精美的高水平课程论文。

## 2. 核心技术栈

* **领域知识：** 相干光通信、克尔效应（Kerr Effect）、自相位调制（SPM）、分步傅里叶法（SSFM）。
* **AI 模型：** ANN（人工神经网络）或 Bi-LSTM。
* **工具链：**
* **MCP 工具：** `brave-search` (调研), `scite` (文献验证), `texflow` (LaTeX编译)。
* **计算环境：** Python 3.x, PyTorch, Pandas, Matplotlib, SciPy。



## 3. 论文逻辑架构（严格遵守）

1. **Abstract:** 摘要，突出非线性补偿的必要性与 AI 方案的增益。
2. **Introduction:** 行业背景 -> 痛点（非线性效应 & DBP 复杂度）-> 本文贡献。
3. **Theoretical Foundation:** 推导 NLSE 方程，解释物理损伤机理。
4. **Proposed Method:** 详细描述神经网络结构、输入输出特征映射。
5. **Results & Discussion:** 重点展示 BER 曲线、星座图对比、复杂度对比。
6. **Conclusion:** 总结研究成果。

## 4. 关键仿真参数规范

在进行数据生成和绘图时，请优先使用以下标准参数（除非另有说明）：

* **调制格式：** 16-QAM
* **符号率：** $32 \text{ GBaud}$
* **光纤长度：** $80 \times 10 \text{ km}$ (800km) 或 $1000 \text{ km}$
* **衰减系数 ($\alpha$)：** $0.2 \text{ dB/km}$
* **色散参数 ($D$)：** $17 \text{ ps/nm/km}$
* **非线性系数 ($\gamma$)：** $1.3 \text{ /W/km}$

## 5. Claude Code 操作指令集

### A. 调研阶段指令

> 执行命令：利用 `brave-search` 检索 2024-2026 年间关于“Reduced complexity ANN for Fiber Nonlinearity Compensation”的论文，提取其神经网络的层数、神经元个数及激活函数类型。

### B. 仿真阶段指令

> 执行命令：编写 Python 脚本。
> 1. 使用 `numpy` 或 `scipy` 实现一个基础的 SSFM 仿真模型。
> 2. 生成受非线性噪声影响的 16-QAM 信号数据集。
> 3. 构建 PyTorch 模型并进行训练。
> 4. 导出 `ber_results.csv` 和 `constellation.png`。
> 
> 

### C. 写作与排版指令

> 执行命令：使用 `texflow` 初始化 LaTeX 项目。
> 1. 将物理模型公式化（使用标准的 LaTeX 语法）。
> 2. 撰写“结果分析”章节，描述 AI 补偿方案相比线性补偿在 $Q$-因子上实现的约 $1.5 \text{ dB}$ 提升。
> 
> 

## 6. 写作风格与约束

* **学术性：** 严禁使用夸张词汇（如 "amazing", "miracle"），统一使用中立语态（如 "It is observed that...", "The simulation results indicate..."）。
* **数学规范：** 所有数学变量、公式必须使用 LaTeX 渲染，如 $\beta_2$, $\gamma$, $P_{in}$。
* **图表引用：** 描述实验结果时，必须明确提及“如图 X 所示”（See Fig. X）。
* **真实性检查：** 所有参考文献必须带有真实的 DOI。

---

## 7. 避坑指南（重要）

1. **拒绝幻觉：** 严禁虚构不存在的论文题目，必须通过 `scite` 验证引用。
2. **单位一致：** 确保所有物理量单位统一（km vs m, dBm vs W）。
3. **代码稳健性：** 生成的 Python 脚本必须包含必要的注释和异常处理。

---

## 8. 协作与知识沉淀规范

1. **进度同步：** 每完成一项关键任务，立即将进度写入 Claude.md §10 并 git 提交。
2. **踩坑记录：** 遇到非平凡问题（调试超过 2 轮），将根因、修复方案、架构决策写入 `SSFM_SOP.md`（或对应子系统的 SOP 文档），确保可复用。
3. **SOP 内容标准：** 核心物理/数学公式 → 代码架构（模块树）→ 关键踩坑（症状 + 根因 + 修复）→ 验证清单。

---

## 9. Git 规范

1. **分支策略：** `main` 只存放稳定代码。日常开发一律新建 `feature/xxx` 分支，由人工审核后合并回 `main`。
2. **小步提交：** 每个逻辑单元完成后立即提交，禁止 mega-commit。提交后及时推送。
3. **禁止操作：** 不得提交敏感信息（密钥、密码）、垃圾文件（缓存、日志）；禁止 `--force` 推送。
4. **必须提交：** `claude.md` 属于受管文件，任何更新必须纳入版本控制。

---

## 10. 项目进度追踪

> 最后更新: 2026-05-13

### 已完成

| 阶段 | 任务 | 状态 | 产物 |
|------|------|------|------|
| A. 调研 | 检索 3 篇后2024年轻量化 ANN 光纤非线性补偿论文 | ✅ 完成 | 3 篇核心论文（DOI 已验证） |
| B1. SSFM 仿真器 | 单信道 16-QAM 光纤传输仿真脚本 | ✅ 完成 | `ssfm_simulator.py` |
| B2. 数据集构建 | 生成 EDC 补偿信号 + 原始发射符号 | ✅ 完成 | `fiber_dataset.npz` (5 功率级别) |
| B2. 验证绘图 | 星座图对比（EDC vs TX） | ✅ 完成 | `initial_test.png`, `power_sweep_overview.png` |
| B3. MLP 补偿器 | 轻量 MLP 非线性补偿器训练 | ✅ 完成 | `train_mlp_nlc.py`, `mlp_nlc_model.pt` |
| B3. 结果评估 | EVM / Q-factor / BER / 星座图 | ✅ 完成 | `mlp_results.csv`, 6 张评估图 |
| 工程化 | Git 管理 + SOP 文档 | ✅ 完成 | `SSFM_SOP.md`, 已推送至 `git@github.com:YAHU2024/ofcPap.git` |
| B4. 性能评估 | BER 曲线、Q-factor 对比、复杂度分析 | ✅ 完成 | `eval_performance_b4.py`, `b4_summary.csv`, 4 张评估图 |
| C. 写作排版 | LaTeX 论文撰写与编译 | ✅ 完成 | `paper.tex`, `paper.pdf` (5 页, IEEE 格式) |

### 待完成

| 阶段 | 任务 | 优先级 |
|------|------|--------|
| -- | （暂无待完成任务） | -- |

### 调研阶段产出（3 篇参考论文）

1. **GDP-KAN** (Optics Express, 2025) — 基于 Kolmogorov-Arnold 网络的非线性补偿器
2. **Operator Learning** (Journal of Lightwave Technology, 2025) — 算子学习框架用于光纤信道建模
3. **MT-NN 简化版** (Optics Express, 2025) — 复杂度感知的轻量 MLP 非线性补偿器（**本项目复现对象**）

### 关键仿真结果摘要

**EDC 基线 (SSFM + 线性补偿):**
- EVM U 型曲线: -4 dBm: 0.167 → -2 dBm: 0.153 (最优) → 0 dBm: 0.171 → +2 dBm: 0.234 → +4 dBm: 0.348
- 相位误差单调递增 (0.057 → 0.336 rad)，确认 SPM 非线性相位噪声主导高功率区域
- 数据集结构: `{X_<power>: EDC 输出符号, Y_<power>: 原始发射符号}` × 5 功率级别

**MLP 非线性补偿 (MT-NN 简化版, 3,346 参数, M=2 记忆抽头):**
- Q-factor 提升: -4 dBm: +8.50 dB → -2 dBm: +15.03 dB → 0 dBm: +23.68 dB → +2 dBm: +31.98 dB → +4 dBm: +38.56 dB
- EVM 改善: 高功率 (+4 dBm) 从 0.348 → 0.0041 (98.8% 降低)
- Q-gain 随功率递增，证实 MLP 主要消除确定性非线性损伤（ASE 噪声不可预测）
- 模型输出: `mlp_nlc_model.pt`, `mlp_results.csv`, 星座图/训练曲线 6 张

**B4 性能评估关键结果:**
- BER 曲线: EDC baseline 在 +4 dBm 时 BER=1.46×10⁻¹（高于 FEC 阈值），MLP-NLC 降至测量极限以下
- Q-factor 增益: 8.50 dB (-4 dBm) → 38.56 dB (+4 dBm)，增益随功率单调递增
- 复杂度: MLP 3,346 参数，3,232 实乘/符号，85.6 ns/符号；仅为标准 DBP 的 ~0.09%
- 直接比特计数: 低功率 MLP 补偿后仍有残余符号误码（测试集 ~10⁴ 比特限制），高功率归零
- 产物: `ber_vs_power.png`, `qfactor_comparison.png`, `qgain_vs_power.png`, `evm_cdf_comparison.png`

---
