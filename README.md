# 初赛

环境：比赛 Linux／Python3.11、DTK26.04、Torch2.9.0+das.opt1.dtk2604、HIP6.3.26093、gfx936／8卡。Torch、NumPy、Transformers、Hub、Pillow、safetensors等使用官方预装版本，缺少的运行、比较及源码构建依赖统一放在 `environment/preliminary_dependencies.zip`。

目录说明：

| 目录 | 作用 |
| --- | --- |
| `assets/` | 统一复现使用的公开首帧图片。 |
| `environment/` | 补充依赖压缩包及版本清单，包含运行、比较和源码重建依赖。 |
| `examples/` | 历史参考、输出示例、比较结果及来源记录。 |
| `tools/` | 根目录 `reproduce.sh` 调用的环境准备、源码编译、推理和比较调度程序。 |
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

脚本默认准备补充依赖，从源码编译InfiniCore、InfiniLM、7个算子的主机库及81个九齿内核，再加载本次构建产物完成FL2VA，最后调用未改动的官方比较程序。无需额外添加 `--from-source`。

`environment/preliminary_dependencies.zip` 保持原样。入口只提取其中的第三方Python依赖和补充构建工具，不提取或加载历史项目扩展、算子库及内核缓存；推理入口会核对本次构建报告、文件SHA和实际扩展加载路径。源码和模型计算代码不因本地或平台运行而另行改写。

8卡作业上限1800秒。首次从空目录开始构建；同源码、同依赖的构建检查点可续跑。退出75表示本份作业尚未完成全流程，保留个人AFS中的构建目录，在下一份作业重复 `bash reproduce.sh`。编译失败不会切换到历史预构建产物；全冷编译加推理能否在单份作业内完成，仍需平台实测。仅构建可执行 `bash reproduce.sh --build-only`。

结果保存在 `$IFAB_AFS_HOME/minimax_preliminary_runs/`，未设置AFS时使用仓库 `work/`。回传 `REPRO_RESULT_ZIP` 对应完整ZIP及完整控制台；没有ZIP时回传 `REPRO_OUTPUT_DIRECTORY` 对应完整目录。构建日志和官方比较输出会同步打印到控制台。

本次修改未在本地试跑。v25已验证的是历史预构建环境下的10步FL2VA与官方比较，不能作为本版从零源码构建已完成的证明。默认比较对象为 `examples/reference_fl2va.mp4` 中的历史同配置PyTorch输出。

默认权重路径 `/public/home/aclsylqidf/shared/MiniMax-H3`，原公开首帧 `assets/Golde33443.jpg`。可用 `--model-path`／`--first-frame` 指定对应实际路径。
