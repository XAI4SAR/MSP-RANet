# Implementation Notes / 实现说明

## 中文

以下信息来自当前源码，涉及论文对齐及历史权重评估时需要核对。

1. **输入通道顺序需要对齐。** `enhance_sar_image_v2()` 输出 OpenCV BGR，即 `[结构张量, Harris, 强度]`。默认 `Format._format_img()` 会反转为 RGB，即 `[强度, Harris, 结构张量]`。但 `DualStreamStem.forward()` 当前仍将张量索引 `2` 当作强度主流，将 `0:2` 当作先验。因此，当前默认路线实际把结构通道送入主流。声称严格复现物理分支前，应统一编码器、格式化与 stem 的约定；仅修改训练时的 `bgr` 会造成训练/验证不一致，修改约定也会影响旧权重的解释。
2. **增强顺序影响物理通道。** 当前先验生成位于 Mosaic 之前，其他 Mosaic 图片通过普通加载取得，尚未生成对应先验；HSV 也会混合物理通道。原始训练脚本未覆盖 Mosaic 和 HSV 参数，沿用源码默认增强设置。
3. **HNR 使用分阶段调度。** 代码基础分类项为 `BCEWithLogitsLoss`；HNR margin 固定为 `1`，前 `100` 轮预热，之后以余弦方式逐渐增加到权重上限 `0.5`。训练 `200` 轮时，从第 `102` 轮才出现正权重；总轮数不超过 `100` 时不启用该项。这与稿件描述的 `λ=1` 不同，默认参数不足以证明严格复现。HNR 已合并进 `cls_loss`，没有单独日志曲线。
4. **数据与评估必须一致。** 核对飞机子集、划分清单、转换后的标签及类别编号；当前仓库未附 FAIR 子集的精确清单。模板先验参数沿用所提供的数据配置，Harris 平滑尺度和响应阈值也需要与稿件核对。训练和测试应使用相同先验设置。
5. **保留实际运行记录。** 建议保存 `args.yaml`、`results.csv`、权重、依赖版本、划分文件及源码修改。若更改图像尺寸、batch、增强或通道约定，应在实验记录中说明。

## English

These details are visible in the supplied source and matter when comparing with the manuscript or evaluating historical weights.

1. **Input channel order needs alignment.** `enhance_sar_image_v2()` returns OpenCV BGR `[structure, Harris, intensity]`. The default `Format._format_img()` reverses this into RGB `[intensity, Harris, structure]`. However, `DualStreamStem.forward()` currently treats tensor index `2` as the intensity stream and indices `0:2` as priors. Thus the existing default route sends the structural channel to the main stream. Reconcile the encoder, formatter and stem conventions before claiming strict physical-branch reproduction. Changing only training `bgr` creates a train/validation mismatch; changes also affect the interpretation of existing checkpoints.
2. **Augmentation placement matters.** Enhancement runs before Mosaic, while additional Mosaic images are loaded without that enhancement. HSV augmentation also mixes physical channels. The original training script does not override Mosaic or HSV parameters and uses the source defaults.
3. **HNR uses a scheduled implementation.** In this code, the base classification term is `BCEWithLogitsLoss`. HNR uses margin `1`, a `100`-epoch warmup and a cosine ramp toward weight `0.5`. At `200` epochs, the first positive weight appears in epoch `102` (one-based). Runs of `100` epochs or fewer never activate the term. This differs from the manuscript's stated `λ=1`, so the defaults alone do not establish an exact reproduction. HNR is included in `cls_loss`; it has no separate log curve.
4. **Data and evaluation must match.** Reproduce the aircraft subset, split files, converted annotations and class-ID mapping. The repository does not supply the exact FAIR subset manifests. The template prior-generation settings follow the supplied dataset configs; also check Harris smoothing and response thresholds against the manuscript. Keep enhancement settings the same for training and evaluation.
5. **Keep the actual run configuration.** Archive `args.yaml`, `results.csv`, checkpoint, package versions, data split and any source changes. Report changes to image size, batch, augmentation or channel conventions alongside results.
