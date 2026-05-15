# Chinese LaTeX Paper Writing SOP —— xelatex + xeCJK + IEEEtran 中文学术论文排版

## Overview

This document records the core logic, code architecture, and pitfalls encountered while adapting the English IEEEtran paper into a Chinese-language academic manuscript suitable for Chinese journal submission (e.g., 通信学报). The primary challenge was integrating CJK (Chinese/Japanese/Korean) font support with the IEEEtran document class, which was designed for English-only typesetting.

---

## 1. Core Problem & Key Decisions

### 1.1 Why xelatex Instead of pdflatex

| Engine | CJK Support | IEEEtran Compat | Decision |
|--------|------------|-----------------|----------|
| `pdflatex` | Requires `CJK` package, limited font support | Native | Rejected — poor Chinese font handling |
| `xelatex` | Native Unicode + `xeCJK` package, system fonts | Works via font fallback | **Selected** |
| `lualatex` | Native Unicode, `luatexja` package | Works | Overkill for this task |

**Key constraint:** IEEEtran hardcodes Times Roman (`ptm`) font metrics for English text via LaTeX's NFSS (New Font Selection Scheme). Under xelatex, `ptm` has no Unicode (TU) encoding, triggering font fallback warnings.

### 1.2 Architecture Decision: xeCJK Over ctex

| Package | Pros | Cons |
|---------|------|------|
| `ctex` | All-in-one Chinese solution, redefines `\abstract` etc. | May clash with IEEEtran's custom `\abstract` definition |
| `xeCJK` | Lightweight, only handles font selection | Requires manual `\setCJKmainfont` configuration |

**Chosen: `xeCJK`** — Minimal intrusion into IEEEtran's macro definitions. The `ctex` package redefines `\abstract`, `\section`, etc., which can conflict with IEEEtran's customized versions.

---

## 2. LaTeX Project Architecture

```
paper_zh_v3.tex
│
├── Preamble (lines 1–17)
│   ├── \documentclass[conference]{IEEEtran}   — IEEE conference format
│   ├── Standard packages (graphicx, amsmath, booktabs, hyperref)
│   ├── \usepackage{xeCJK}                     — CJK font management
│   ├── \setCJKmainfont{SimSun}                — Serif: 宋体
│   ├── \setCJKsansfont{SimHei}                — Sans-serif: 黑体
│   └── \setCJKmonofont{SimSun}                — Monospace: 宋体
│
├── Title Block (lines 19–35)
│   ├── \title{...}                            — Chinese title
│   ├── \author{\IEEEauthorblockN{...}         — Author name (中文)
│   │         \IEEEauthorblockA{...}}          — Affiliation (中文)
│   ├── \maketitle
│   └── 中图分类号 (CLC number for Chinese journals)
│
├── Body Sections
│   ├── \begin{abstract}...\end{abstract}      — Chinese abstract
│   ├── §1. 引言 (Introduction)
│   ├── §2. 理论基础 (Theoretical Foundation)
│   │   ├── §2.1 非线性薛定谔方程
│   │   ├── §2.2 对称分步傅里叶法
│   │   ├── §2.3 掺铒光纤放大器与ASE噪声
│   │   ├── §2.4 电子色散补偿
│   │   └── §2.5 性能指标
│   ├── §3. 所提方法 (Proposed Method)
│   │   ├── §3.1 系统模型与仿真设置
│   │   └── §3.2 MLP非线性补偿器架构
│   ├── §4. 结果与讨论 (Results & Discussion)
│   │   ├── §4.1 星座图与EVM分析
│   │   ├── §4.2 Q因子与BER性能
│   │   ├── §4.3 计算复杂度
│   │   └── §4.4 讨论
│   └── §5. 结论 (Conclusion)
│
├── English Block (lines 322–333)
│   ├── \section*{English Title, Abstract and Keywords}
│   └── Required by Chinese journals for indexing
│
└── References (lines 335–356)
    ├── \begin{thebibliography}{10}
    └── GB/T 7714 format (Chinese national standard)
```

---

## 3. Compilation Workflow

### 3.1 Compilation Command

```bash
# MUST use xelatex (NOT pdflatex)
xelatex -interaction=nonstopmode -shell-escape paper_zh_v3.tex

# Run twice for cross-references
xelatex -interaction=nonstopmode -shell-escape paper_zh_v3.tex
```

**Why `-shell-escape`:** Required for `xeCJK` to auto-detect system fonts on Windows.

### 3.2 Font Requirements (Windows)

| Command | Font Name | File | Purpose |
|---------|-----------|------|---------|
| `\setCJKmainfont{SimSun}` | 宋体 | `simsun.ttc` | Body text |
| `\setCJKsansfont{SimHei}` | 黑体 | `simhei.ttf` | Section headings |
| `\setCJKmonofont{SimSun}` | 宋体 | `simsun.ttc` | Monospace fallback |

**Verification command:**
```bash
fc-list :lang=zh | grep -i -E "sim|song|hei"
```

---

## 4. Pitfalls & Fixes

### Pit 1: "Missing $ inserted" at `\noindent` (Critical)

**Symptom:**
```
! Missing $ inserted.
<inserted text> $
l.24 \noindent
```

**Root cause:** Using `_` (underscore) characters in Chinese text outside of math mode. The underscore `_` is a LaTeX math-mode subscript character. When Chinese text was pasted into the `.tex` file, some characters or formatting included invisible or mis-encoded underscores that triggered the error at the first non-math command encountered (typically `\noindent` or `\begin{abstract}`).

**Debugging procedure:**
1. First, check the line mentioned in the error — but since `\noindent` itself is harmless, the actual cause is earlier in the file
2. Comment out the abstract and gradually uncomment to isolate the offending character
3. Search for bare `_` outside math mode: `grep -n '_' paper.tex` and verify each is inside `$...$` or `\begin{equation}`

**Fix:** Ensure ALL underscores appear only within math mode (`$...$` or equation environments). In Chinese text, replace any `_` with its full-width equivalent or re-encode the Chinese text block.

**Prevention:** Use `_test_pkg.tex` (a minimal template with just `\usepackage{xeCJK}` + a test character) to validate the CJK setup BEFORE inserting full paper content.

---

### Pit 2: IEEEtran Font Warnings with xelatex

**Symptom:**
```
LaTeX Font Warning: Font shape `TU/ptm/m/n' undefined
(Font)              using `TU/lmr/m/n' instead
```

**Root cause:** IEEEtran selects Times (`ptm`) via `\fontfamily{ptm}\selectfont`. Under xelatex with Unicode (TU) encoding, the `ptm` font has no TU-encoded variant, so Latin Modern (`lmr`) is substituted. The Chinese fonts (SimSun/SimHei) are unaffected because `xeCJK` manages them independently.

**Fix:** These are cosmetic warnings only. The output PDF renders correctly because:
- English text: Latin Modern substitutes for Times (visually similar)
- Chinese text: Rendered via xeCJK with SimSun (correct)
- Math: Uses standard Computer Modern math fonts (correct)

**DO NOT attempt to fix this** by loading `fontspec` with `\setmainfont{Times New Roman}`, as this will break IEEEtran's carefully tuned line-spacing and float placement calculations.

---

### Pit 3: IEEEtran `[h]` Float Specifier Conflict

**Symptom:**
```
LaTeX Warning: `h' float specifier changed to `ht'.
```

**Root cause:** IEEEtran enforces stricter float placement than standard LaTeX. The `[h]` (here) specifier is internally converted to `[ht]` (here-or-top) because IEEEtran never places floats at the exact text position in two-column mode.

**Fix:** Accept this as normal behavior. IEEEtran's float algorithm is designed for two-column academic publishing. For the Chinese paper, figures and tables will naturally float to column tops — this is the expected layout.

---

### Pit 4: `\noindent\textbf{中图分类号：}` Produces Extra Space

**Symptom:** The CLC number line `\noindent\textbf{中图分类号：}TN913.7` produced unexpected vertical spacing before the abstract.

**Root cause:** `\noindent` starts a new paragraph, and IEEEtran's abstract environment has built-in spacing. The `\vspace{2mm}` command interacts unpredictably with IEEEtran's internal spacing.

**Fix:** Place `\noindent\textbf{中图分类号：}TN913.7` AFTER `\maketitle` but BEFORE `\begin{abstract}`, and use `\vspace{2mm}` only if the spacing is actually insufficient in the compiled PDF.

---

### Pit 5: GB/T 7714 Reference Format Conversion

**Symptom:** Converting IEEE reference format to GB/T 7714 (Chinese national standard) was tedious and error-prone to do manually.

**Root cause:** IEEE and GB/T 7714 disagree on: author name casing, punctuation between fields, journal name abbreviation, and volume/issue notation.

**Key format differences:**

| Element | IEEE | GB/T 7714 |
|---------|------|-----------|
| Author name | E. Ip, J. M. Kahn | IP E, KAHN J M |
| Title case | "Compensation of Dispersion..." | Same as original |
| Journal name | J. Lightw. Technol. | Journal of Lightwave Technology |
| Year position | Oct. 2008 | 2008 |
| Volume/Issue | vol. 26, no. 20 | 26(20) |
| Pages | pp. 3416--3425 | 3416-3425 |
| Periodical type tag | None | [J] for journal, [M] for monograph |

**Fix:** Write references manually in `thebibliography` environment rather than using BibTeX, since Chinese journals with GB/T 7714 require specific formatting not supported by standard `.bst` files.

---

## 5. Iteration History & Debugging Path

| Version | Engine | Status | Key Issue |
|---------|--------|--------|-----------|
| `_test_pkg` | xelatex | OK (warnings only) | Verified xeCJK + IEEEtran compatibility; font fallback warnings are cosmetic |
| `_test2` | xelatex | FAILED | "Missing $" at `\noindent` — bare underscore in pasted Chinese text |
| `_test3` | xelatex | OK | Clean template, no CJK content yet |
| `_test4` | xelatex | OK | Added Chinese abstract — verified xeCJK works with IEEEtran abstract env |
| `_test5` | xelatex | FAILED | "Missing $" recurrence — second instance of unescaped special char |
| `paper_zh_v2` | xelatex | OK | Full paper, author: YaHu / 西安交通大学 |
| `paper_zh_v3` | xelatex | OK | Final: updated author info to 黄子健 / 武夷学院海峡成功学院 |

**Total: 7 compilations, 2 failures (both "Missing $" from unescaped characters).**

---

## 6. Validation Checklist

After compiling `paper_zh_v3.tex` with xelatex (×2), verify:

| Check | Expected Result | How to Verify |
|-------|----------------|---------------|
| No fatal errors | Clean exit, PDF produced | `grep -c "!" paper_zh_v3.log` = 0 |
| Chinese characters render | All CJK text visible, no tofu (□) | Open PDF, scan sections 1-5 |
| English block renders | Title/Abstract/Keywords in English section | Check last page |
| Figures display | 4 figures (overview, Q-factor, BER, Q-gain) | `grep "figure" paper_zh_v3.log` |
| References display | 5 references in GB/T 7714 format | Check reference section |
| Cross-references resolved | No "??" in figure/table citations | Search PDF for "??" |
| Font warnings only | Font shape warnings expected, no other warnings | `grep -i warning paper_zh_v3.log` |
| IEEEtran two-column layout | 5+ pages, balanced columns | Visual inspection |

---

## 7. Key Design Decisions

1. **xelatex over pdflatex** — Mandatory for Unicode CJK support. pdflatex's `CJK` package is legacy and font-limited.

2. **xeCJK over ctex** — ctex's macro redefinitions (`\abstract`, `\section`) clash with IEEEtran's internal macros. xeCJK is a lighter touch.

3. **SimSun/SimHei over Noto CJK** — Windows system fonts are always available, avoiding the need to bundle ~15MB font files with the project.

4. **Manual GB/T 7714 formatting** — No reliable BibTeX style file exists for GB/T 7714 within standard TeX Live. Manual `thebibliography` entries are more maintainable than a custom `.bst` for a 5-reference paper.

5. **English block at end** — Chinese journals (通信学报, etc.) require an English title/abstract/keywords section for international indexing databases.

6. **Separate Chinese `.tex` file from English `paper.tex`** — The LaTeX engines differ (xelatex vs pdflatex), the font setup is completely different, and the reference format differs. Maintaining separate source files avoids conditional compilation complexity.
