# 第6章理论：非线性优化基础

## 1. 物理概念

SLAM 后端目标是最小化观测误差，核心流程是“线性化-求增量-迭代更新”。

## 2. 数学公式

$$
\min_{\mathbf{x}} \sum_i \|\mathbf{e}_i(\mathbf{x})\|^2
$$

Gauss-Newton 法方程：

$$
\mathbf{H}\Delta\mathbf{x}=\mathbf{b},\quad
\mathbf{H}=\sum_i\mathbf{J}_i^\top\mathbf{J}_i,\quad
\mathbf{b}=-\sum_i\mathbf{J}_i^\top\mathbf{e}_i
$$

## 3. 代码映射

- `code/ch6/gaussNewton.cpp`

## 4. 深入阅读

- [非线性优化（专题版）](./nonlinear-optimization.md)
