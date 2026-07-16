# Temporal independent v3

这个目录最初作为完全独立的平行实验建立，不修改 `research/` 主训练流程。当前实验已通过跨 split 验证并被接入正式网页后端，但训练与汇总仍保持独立，旧 Cox 流程可以单独运行和回退。

## 目标

当前多个 GNN/MLP/CTM 风险分数高度相关，继续叠加同类 residual 很难突破现有 C-index。这个实验同时改变两件事：

1. 保留每一条菌群边的身份、方向和节点状态，构造 topology fingerprint，避免只使用全图均值和方差。
2. 使用 XGBoost AFT 生存目标学习完整生存时间分布，并正确处理右删失样本，而不是再训练一个相似的 Cox residual。

最终仍以现有主模型为 reference。融合权重只在 validation 上选择，并允许 `alpha=0`，因此弱专家可以被完整拒绝。

## 初始运行

```powershell
python -m experiments.temporal_independent_v3.topology_aft_fusion `
  --config research_config_v2.yaml `
  --mainline-predictions outputs/current_mainline_v2/full_risk_head_refiner_v2/split42_three_seed_summary_selected_predictions.npz `
  --split-seed 42 `
  --seed 7 `
  --output-dir outputs/current_mainline_v2/temporal_independent_v3/split42_seed7
```

输出包括候选 AFT 模型、完整预测数组和 `summary.json`。只有 validation C-index 达到最低增益时，融合结果才会替代 reference。

## 当前结果

固定使用 seed `7, 21, 42, 123, 2026`，先对 AFT 风险取均值，再通过 split 42/43 的 validation 曲线选择同一个共识权重。共识规则保留每个 split 至少 95% 的验证峰值增益，全程不使用 test 标签选权重。

| 指标 | 当前主线 | Temporal topology v3 | 变化 |
|---|---:|---:|---:|
| 两 split 平均 test C-index | 0.740333 | 0.757056 | +0.016723 |
| 平均校准 Cox loss 变化 | - | - | -0.013212 |
| C-index 改善的 split | - | 2/2 | - |
| Cox loss 改善的 split | - | 2/2 | - |

共识权重为 `alpha=0.63`。split 42 达到 `0.760866`，split 43 达到 `0.753247`。

消融实验显示，旧式节点/全图摘要配合 AFT 只能带来小幅提升；加入 10 条有身份和方向的具体边后，增益扩大到约 `+0.015 ~ +0.019`。因此主要新信息来自 edge identity，完整时间监督是辅助贡献。

最终汇总位于：

```text
outputs/current_mainline_v2/temporal_independent_v3/cross_split_consensus/cross_split_consensus_summary.json
```

注意：当前 `topology_v6` 是 synthetic/noisy augmented 数据。该结果证明方法在现有实验协议中有效，但不能替代外部真实队列验证。

网页端还存在一个额外边界：研究评估使用表内拓扑和确定性的 8 样本上下文，网页端则从可提交字段推断拓扑，并使用固定校准 anchor 保证单样本推理不依赖并发请求。因此 `0.757056` 是正式离线协议结果，不能直接表述为网页部署模型的临床 C-index。
