# 初赛

环境：比赛 Linux／Python3.11、DTK26.04、Torch2.9.0+das.opt1.dtk2604、HIP6.3.26093、gfx936／8卡。Torch、NumPy、Transformers、Hub、Pillow、safetensors等使用官方预装版本，缺少的运行、比较及源码构建依赖统一放在 `environment/preliminary_dependencies.zip`。

目录说明：

| 目录 | 作用 |
| --- | --- |
| `assets/` | 统一复现使用的公开首帧图片。 |
| `environment/` | 补充依赖压缩包及版本清单，包含运行、比较和源码重建依赖。 |
| `examples/` | 历史参考、输出示例、比较结果及来源记录。 |
| `tools/` | 根目录 `reproduce.sh` 调用的环境准备、推理和比较调度程序。 |
| `source_rebuild/` | 源码重建入口及所需源码、配置和运行支持。 |
| `source_rebuild/config/` | 构建参数、运行配置和平台依赖要求。 |
| `source_rebuild/runtime/` | 模型初始化兼容、算子接入、运行观察及输出检查代码。 |
| `source_rebuild/tools/` | 构建产物定位、运行环境配置及平台接口辅助代码。 |
| `source_rebuild/source/` | InfiniCore、InfiniLM原始源码快照与本项目改动。 |
| `source_rebuild/source/patches/` | 对InfiniCore、InfiniLM源码快照应用的补丁。 |
| `source_rebuild/source/operators/` | 各算子的九齿生成器、主机接入代码和构建配置。 |
| `source_rebuild/source/operators/cast/` | 类型转换算子源码。 |
| `source_rebuild/source/operators/rotary/` | H3旋转位置编码算子源码。 |
| `source_rebuild/source/operators/mrope/` | 多模态旋转位置编码算子源码。 |
| `source_rebuild/source/operators/silu/` | SiLU激活算子源码，包括视频VAE接入。 |
| `source_rebuild/source/operators/swiglu/` | SwiGLU激活算子源码。 |
| `source_rebuild/source/operators/rms/` | RMSNorm归一化算子源码。 |
| `source_rebuild/source/operators/mha/` | 多头注意力算子源码。 |

在仓库根目录运行：

```bash
git lfs pull
bash reproduce.sh
```

脚本默认解压并配置仓库私有环境、加载预构建原生库及81个九齿内核、检查并完成完整FL2VA。

默认权重路径 `/public/home/aclsylqidf/shared/MiniMax-H3`，原公开首帧 `assets/Golde33443.jpg`。可用 `--model-path`／`--first-frame` 指定对应实际路径。