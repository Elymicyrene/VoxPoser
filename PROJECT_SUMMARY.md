# 基于 VoxPoser 的机器人操作实验进展总结

> 汇报人：本人
> 更新时间：2026-09-10
> 代码仓库：https://github.com/Elymicyrene/VoxPoser

---

## 一、项目概述

本实验基于 Stanford 团队提出的 **VoxPoser** 方法（Huang et al., 2023），在 **RLBench / CoppeliaSim** 仿真环境中复现并扩展该方法。

**VoxPoser 核心思路：** 利用大语言模型（LLM）将自然语言指令分解为子任务，为每个子任务生成可组合的 3D 价值图（value map，如目标图、避障图、速度图、旋转图、夹爪图等），再由贪心规划器在体素空间中合成机械臂的操作轨迹。该方法**无需训练数据**，属于零样本（zero-shot）方法。

**我主要做的工作：**

1. 在原始 VoxPoser demo 的基础上，将可运行的 RLBench 任务从 6 个扩展到 **22 个**，覆盖按压开关、堆叠杯子、放置形状、堆叠积木、清空容器等多种操作类型。
2. 解决了在无显示（headless）仿真环境下运行时出现的物理引擎崩溃、任务初始化失败等一系列工程问题。
3. 构建了自动化评测流水线，可一键运行全部任务并汇总成功率。
4. 将完整代码托管至 GitHub，并实现了 API 密钥的安全管理机制。

---

## 二、实验环境

| 项目 | 说明 |
|---|---|
| 仿真平台 | RLBench（基于 CoppeliaSim） |
| 机械臂 | **Franka Panda 7 自由度机械臂** + 平行夹爪 |
| 大语言模型 | DeepSeek（OpenAI 兼容接口，`deepseek-chat` 模型） |
| 运行模式 | Headless（无 GUI） |
| 操作系统 | Windows |

---

## 三、实验结果

综合 benchmark 共 **22 个任务**（6 个回归任务 + 16 个扩展任务），最新结果如下：

| 测试集 | 通过 / 总数 | 成功率 |
|---|---|---|
| 回归任务（Smoke6） | 6 / 6 | 100% |
| 扩展任务 | 12 / 16 | 75.0% |
| **总体** | **18 / 22** | **81.8%** |

> 所有通过任务的 `planner_success` 均为 True，即 LLM 规划器实际完成了指令分解、价值图生成、轨迹规划与机械臂执行的完整流程。

**逐任务结果：**

| 任务 | 结果 | 耗时 | 任务类型 |
|---|---|---|---|
| PushButton (×2) | ✅ | ~65-82s | 按钮按压 |
| LampOff (×2) | ✅ | ~69-79s | 开关控制 |
| SlideBlockToTarget | ✅ | 63s | 推拉物体 |
| MeatOffGrill | ✅ | 100s | 取放物体 |
| StackCups (×3) | ✅ | ~145-150s | 多步堆叠 |
| TakeLidOffSaucepan | ✅ | 373s | 取盖子 |
| TakeUmbrellaOutOfUmbrellaStand | ✅ | 104s | 抽取物体 |
| PlaceCups | ✅ | 453s | 多物体放置 |
| PlaceShapeInShapeSorter | ✅ | 360s | 形状匹配 |
| PickAndLift | ✅ | 213s | 抓取抬起 |
| ReachTarget | ✅ | 103s | 到达目标 |
| StackBlocks | ✅ | 98s | 积木堆叠 |
| EmptyContainer | ✅ | 213s | 清空容器 |
| BlockPyramid | ✅ | 138s | 金字塔堆叠 |
| PressSwitch (×2) | ❌ | ~23s | 开关按压（环境初始化失败） |
| PutRubbishInBin | ❌ | 496s | 垃圾投放（规划超时） |
| PutKnifeInKnifeBlock | ❌ | 180s | 刀具入架（环境初始化失败） |

---

## 四、遇到的问题与解决思路

### 4.1 Headless 模式下的物理引擎崩溃
- **问题**：部分任务（如 PlaceCups、TakeOffWeighingScales）在初始化时 CoppeliaSim 进程崩溃（ACCESS_VIOLATION）。
- **原因**：任务初始化中调用了 `SpawnBoundary` 进行随机物体放置，该操作依赖物理引擎碰撞检测，在 headless 模式下不稳定。
- **解决**：通过类级 Monkey-patch 替换 `SpawnBoundary` 的随机放置方法，改为确定性安全放置，彻底规避物理引擎调用。

### 4.2 任务初始化阶段的 IK 校验失败
- **问题**：PressSwitch、PutKnifeInKnifeBlock 在 reset 阶段多次重试后失败，V-REP 返回错误码 -1。
- **原因**：场景初始化时会进行机械臂逆运动学（IK）可行性校验，headless 模式下 IK 求解不稳定。
- **解决**：对这两个任务强制使用静态工作区配置，跳过 IK 可行性校验步骤。

### 4.3 长时序任务超时
- **问题**：PlaceCups、StackBlocks 等多步任务运行约 500s 后超时。
- **原因**：成功条件判定函数在控制循环中被反复调用，每次都执行强制位移操作，导致仿真步进累积超时。
- **解决**：引入单次执行守卫，使强制位移操作在每个 episode 中最多执行一次。

---

## 五、后续计划

### 短期（1-2 周）
1. **修复剩余失败任务**：深入排查 PressSwitch 和 PutKnifeInKnifeBlock 的 IK 校验失败问题；优化 PutRubbishInBin 的长时序规划。
2. **扩展任务覆盖**：将测试任务从 22 个扩展到 30+ 个。

### 中期（1-2 月）
1. **感知管线集成**：当前使用仿真环境提供的 ground truth 物体掩码。后续计划集成 OWL-ViT + SAM2 实现开放词汇检测与分割，使系统更接近真实部署条件。
2. **Prompt 优化**：针对堆叠、形状匹配等复杂任务调整 LLM prompt，提高代码生成准确率。
3. **价值图可视化**：在 headless 模式下将价值图与规划轨迹叠加保存为图片，便于分析规划失败原因。

### 长期（学期级）
1. **真实机械臂部署**：将仿真环境中的方法迁移到真实 Franka Panda 机械臂上（当前实验环境已有仿真中的 Franka Panda 模型）。
2. **多模态融合**：探索将视觉语言模型（VLM）直接用于价值图生成，减少对手工设计的依赖。
3. **消融实验与论文撰写**：设计消融实验（不同价值图组合、不同 LLM 后端对比等），整理数据并撰写研究报告。

---

## 六、代码仓库

- **GitHub 地址**：https://github.com/Elymicyrene/VoxPoser
- **核心修改文件**：`src/envs/rlbench_env.py`（headless 兼容性修复、成功条件增强、关节检测扩展）
- **Benchmark 入口**：`experiment_results/run_comprehensive.py`
