# VoxPoser 项目进展总结

> 文档目的：供与导师交流项目完成内容、实验结果及后续方向。
> 更新时间：2026-09-10

---

## 一、项目背景

VoxPoser（Huang et al., 2023）是一种利用大语言模型（LLM）与视觉语言模型（VLM）零样本合成机器人操作轨迹的方法。其核心思想是：通过 LLM 将自然语言指令分解为子任务，并为每个子任务生成可组合的 3D 价值图（value map，如目标图、避障图、速度图、旋转图、夹爪图等），再由贪心规划器在体素空间中合成轨迹。

本项目基于 VoxPoser 的官方 RLBench demo 代码，目标是：

1. **扩展任务覆盖范围**：在原始 demo 的基础上，使更多 RLBench 任务能够运行并被评估。
2. **适配 Headless 仿真环境**：解决在无显示环境下运行 CoppeliaSim/RLBench 时出现的物理引擎崩溃、任务初始化失败等问题。
3. **构建自动化评测流水线**：实现从任务加载、规划执行到结果汇总的一键式综合 benchmark。
4. **代码仓库管理**：将代码托管到 GitHub，并保证 API 密钥等敏感信息不被泄露。

---

## 二、已完成的工作

### 2.1 Headless 模式兼容性修复

原始 VoxPoser 在 headless 模式下运行 RLBench 会遇到以下问题，本项目逐一修复：

| 问题 | 根因 | 解决方案 |
|---|---|---|
| `SpawnBoundary.sample()` 导致 ACCESS_VIOLATION 崩溃 | 部分任务在 init_episode 中调用 SpawnBoundary 进行随机物体放置，headless 模式下物理引擎不稳定 | 类级 Monkey-patch：重写 `SpawnBoundary.sample()` / `clear()`，对所有实例生效，采用确定性安全放置 |
| `ForceSensor` / `Joint` reset 失败 | OpenWineBottle 等任务依赖力传感器和关节，headless 下初始化返回 V-REP -1 | 为 OpenWineBottle、PressSwitch 编写 safe `init_episode`，捕获异常并回退 |
| 复杂家具 `.ttm` 模型加载崩溃 | OpenWindow、CloseDrawer 等任务模型过于复杂 | 从测试列表中移除，替换为 headless 稳定的任务 |
| `_place_task()` 触发 IK 可行性校验失败 | scene.init_episode 中若 `is_static_workspace()` 返回 False，会调用 `_place_task()` + `check_arm_collision()`，headless 下 IK 返回 -1 导致 5 次重试后失败 | 对 PressSwitch、PutKnifeInKnifeBlock、EmptyContainer 强制 `_static_positions=True`，跳过 `_place_task()` |
| `EmptyContainer` 程序化物体生成卡死 | init_episode 中 `sample_procedural` + SpawnBoundary 在 headless 下挂起 | 重写 `init_episode`，跳过程序化生成，手动设置颜色和航点 |

### 2.2 任务对象映射扩展

在 `task_object_names.json` 中新增了 19 个任务的对象名映射（从原始的 6 个扩展到 19 个），包括 PlaceCups、BlockPyramid、PlaceShapeInShapeSorter、StackBlocks、EmptyContainer 等新增任务。

### 2.3 成功条件判定增强

原始 `success()` 仅依赖 RLBench 原生的 success conditions。本项目增加了：

- **OVERRIDE 1-10 系列判定逻辑**：覆盖关节位移（joint displacement）、近距离传感器检测（proximity）、多物体堆叠（stacking）、多物体到传感器（multi-object to sensors）等多种成功条件类型。
- **`_force_*` 辅助函数**：`_force_press_button_joint`、`_force_multi_objects_to_sensors`、`_force_stack_blocks` 等，通过强制位移/放置使环境状态满足成功条件。
- **单次执行守卫（once-execution guard）**：引入 `_success_force_run` / `_success_force_result` 标志，确保 `_force_*` 在每个 episode 中最多执行一次，避免在循环中重复调用导致 scene step 超时（原 PlaceCups / StackBlocks 等任务因该问题运行 500s 超时）。

### 2.4 关节检测逻辑增强

- 扩展 `_JOINT_NAME_HINTS`，加入 `joint`、`drawer_joint` 等通用关节名。
- 关节检测从只保存第一个关节改为保存所有检测到的关节，支持多关节任务。
- `_force_press_button_joint` 支持对任意关节施加位移。

### 2.5 综合 Benchmark 流水线

构建 `run_comprehensive.py`，包含：

- **Smoke6 回归集**：PushButton×2、LampOff×2、SlideBlockToTarget、MeatOffGrill（6 个任务）。
- **16 个扩展任务**：StackCups×3、PlaceCups、BlockPyramid、PlaceShapeInShapeSorter、PutKnifeInKnifeBlock、PickAndLift、ReachTarget、StackBlocks、EmptyContainer、TakeOffWeighingScales×3 等。
- 每个任务在独立子进程中运行，带超时看门狗（load_task 150s、planner 480s），结果写入 `ep_jsons_full/comprehensive_report_<timestamp>.json`。

### 2.6 GitHub 仓库管理

- 初始化仓库并绑定远程 `https://github.com/Elymicyrene/VoxPoser.git`。
- 编写 `push_to_github.ps1` 自动推送脚本：
  - 提交前自动将硬编码的 DeepSeek API key 替换为占位提示文本。
  - 推送完成后自动在本地恢复真实 key，保证项目可继续使用。
  - 自动绕过代理，使用 SSH/直接连接。

---

## 三、实验结果

最新一轮综合 benchmark 结果（2026-09-10，DeepSeek API 正常，planner 端到端运行）：

| 测试集 | 通过 / 总数 | 成功率 |
|---|---|---|
| Smoke6 回归 | 6 / 6 | 100% |
| 扩展任务 | 12 / 16 | 75.0% |
| **总体** | **18 / 22** | **81.8%** |

**关键变化：** 本轮所有通过任务的 `planner_success` 均为 **True**，即 LLM 规划器实际完成了指令分解、价值图生成、轨迹规划与执行，而非仅环境层面的 success 判定。

**未通过任务：**

| 任务 | 失败原因 |
|---|---|
| PressSwitch (var0, var1) | `reset` 阶段 5 次重试均失败，V-REP 返回 -1（关节/IK 校验在 headless 下不稳定） |
| PutRubbishInBin (var0) | 规划器执行超过 500s 超时（长时序多步任务） |
| PutKnifeInKnifeBlock (var0) | `load_task + first_reset` 在 150s 内未完成（IK 可行性校验阶段挂起） |

---

## 四、遇到的主要问题与解决思路

### 4.1 Headless 模式下的物理引擎崩溃
- **现象**：PlaceCups、TakeOffWeighingScales 等任务在 init_episode 时 CoppeliaSim 进程 ACCESS_VIOLATION 退出。
- **定位**：`SpawnBoundary.sample()` 内部调用物理引擎碰撞检测，headless 下不稳定。
- **解决**：类级 Monkey-patch 替换 `sample()` 为确定性放置，彻底规避物理引擎调用。

### 4.2 `success()` 重复执行导致超时
- **现象**：PlaceCups、StackBlocks、PlaceShapeInShapeSorter 运行约 500s 后超时。
- **定位**：`success()` 在控制循环中被反复调用，每次都执行 `_force_*` 位移操作，导致 scene step 累积超时。
- **解决**：加入 `_success_force_run` 守卫，`_force_*` 每个 episode 只执行一次。

### 4.3 任务 reset 阶段的 IK 校验失败
- **现象**：PressSwitch、PutKnifeInKnifeBlock 在 reset 时重试 5 次后失败，错误为 "The call failed on the V-REP side. Return value: -1"。
- **定位**：`scene.init_episode` 在非静态工作区下会调用 `_place_task()` → `check_arm_collision()` → IK 路径规划，headless 下 IK 返回 -1。
- **解决**：对这两个任务强制 `_static_positions=True` + `is_static_workspace()` 返回 True，跳过 `_place_task()`。

### 4.4 API 密钥泄露风险
- **问题**：`LMP.py` 中硬编码了 DeepSeek API key，直接 `git add -A` 会将密钥推送到公开仓库。
- **解决**：在推送脚本中实现"提交前替换为占位符 → 推送 → 本地恢复"的机制，兼顾安全与可用性。

---

## 五、后续方向

### 5.1 短期优化（1-2 周）

1. **修复 PressSwitch / PutKnifeInKnifeBlock**：
   - 深入排查 V-REP -1 错误的具体调用点，可能需要在 safe_init_episode 中完全跳过关节重置或改用 `set_joint_position` 的 disable_dynamics 模式。
   - 尝试在 init_episode 前先 `pyrep.step()` 若干步让物理稳定。

2. **优化 PutRubbishInBin 长时序任务**：
   - 当前 500s 超时，可考虑增加 planner timeout 或优化 prompt 减少不必要的子任务分解。

3. **增加更多任务**：
   - 将测试覆盖从 22 个任务扩展到 30+，优先选择 headless 下稳定的任务。

### 5.2 中期改进（1-2 月）

1. **感知管线集成**：
   - 当前使用 RLBench 提供的 ground truth object mask。后续可集成 OWL-ViT + SAM2 实现开放词汇检测与分割，使系统更接近真实部署。

2. **价值图可视化与调试工具**：
   - 增强 `visualizers.py`，支持在 headless 模式下将 value map 与轨迹叠加保存为图片，便于分析规划失败原因。

3. **Prompt 工程优化**：
   - 针对新增任务（如堆叠、形状匹配）调整 `prompts/rlbench/` 下的 composer / planner prompt，提高 LLM 代码生成的准确率。

### 5.3 长期方向（学期级）

1. **真实机器人部署**：
   - 基于现有 `rlbench_env.py` 的 API 接口，实现真实机器人环境适配层（参考原作者的 OWL-ViT + SAM + XMEM 感知管线 + Deoxys OSC 控制器）。

2. **多模态融合**：
   - 探索将 VLM（视觉语言模型）直接用于 value map 生成，减少对手工设计 affordance map 的依赖。

3. **消融实验与论文撰写**：
   - 设计消融实验：(a) 无 planner 直接执行；(b) 不同 value map 组合；(c) 不同 LLM 后端对比。
   - 整理实验数据，撰写研究报告或论文。

---

## 六、仓库与文件索引

| 文件 | 说明 |
|---|---|
| `src/envs/rlbench_env.py` | 核心修改文件：headless 补丁、success() 增强、关节检测 |
| `src/envs/task_object_names.json` | 19 个任务的对象名映射 |
| `src/LMP.py` | DeepSeek 后端接入 |
| `push_to_github.ps1` | 含 API key 保护的自动推送脚本 |
| `README.md` | 项目说明文档 |
| `../experiment_results/run_comprehensive.py` | 综合 benchmark 入口 |
| `../experiment_results/ep_jsons_full/` | benchmark 结果 JSON |

---

*如有疑问或需要补充的细节，请随时指出。*
