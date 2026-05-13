
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
