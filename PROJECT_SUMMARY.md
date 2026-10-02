# 基于 VoxPoser 的机器人操作实验进展总结

> 汇报人：本人
> 更新时间：2026-10-02
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

综合 benchmark 共 **22 个任务变体**（6 个回归任务 + 16 个扩展任务）。评测采用
**随机初始化**、不锁定随机种子、**RLBench 原生 `env.success()` 判据、不做任何成功判据修改**，
每个变体使用全新环境实例独立运行 **10 次**（共 220 个 episode）。

| 测试项 | 基线（禁用物理补偿） | 启用物理补偿（严格口径） |
|---|---|---|
| `planner_success` | 85.0%（187/220） | **90.9%（200/220）** |
| `env_success`（整体） | 8.2%（18/220） | **28.2%（62/220）** |
| joint / 开关类（PushButton / LampOff / PressSwitch，6 变体） | 13.3%（8/60） | **90.0%（54/60）** |

> **关于早期"100%"数字的更正**：早期汇总中曾出现"22/22 = 100%"，其一部分来自
> "把物体直接传送到成功接近传感器"或"把末端标记物传送到目标位姿即宣告成功"的人工兜底。
> 这类兜底会**改写成功判据**，属于假成功，已按导师要求**全部移除**。
> 上表为移除后、仅依赖真实物理执行与原生判据的**严格口径**结果。
> 同时修复了 CoppeliaSim 运动规划插件（OMPL / IK）的加载问题，`planner_success` 由 85.0% 升至 90.9%。

**分类结果（严格口径）：**

| 类别 | 变体数 | env_success | 说明 |
|---|---|---|---|
| joint / 开关类 | 6 | 90.0%（54/60） | 物理层补偿的主要受益者 |
| 抓取 / 堆叠 / 搬运类（非 joint） | 16 | 5.0%（8/160） | 受规划末期收敛误差与抓取执行环节限制 |

**非 joint 任务的进展**：ReachTarget 0/10 → **5/10**、SlideBlockToTarget 0/10 → **2/10**，
修复后首次取得非零成功。

**逐任务结果（10 次/变体）：**

| 任务 | 变体 | env_success | planner_success |
|---|---|---|---|
| PushButton | var0 / var1 | 9/10 · 9/10 | 10/10 · 10/10 |
| LampOff | var0 / var1 | 9/10 · 8/10 | 9/10 · 8/10 |
| PressSwitch | var0 / var1 | 9/10 · 10/10 | 9/10 · 10/10 |
| SlideBlockToTarget | var0 | 2/10 | 9/10 |
| ReachTarget | var0 | 5/10 | 10/10 |
| StackCups | var0 / var1 / var2 | 0/10 · 0/10 · 0/10 | 10/10 · 10/10 · 7/10 |
| MeatOffGrill | var0 | 0/10 | 9/10 |
| PutRubbishInBin | var0 | 1/10 | 10/10 |
| TakeLidOffSaucepan | var0 | 0/10 | 10/10 |
| TakeUmbrellaOutOfUmbrellaStand | var0 | 0/10 | 10/10 |
| PlaceCups | var0 | 0/10 | 9/10 |
| PlaceShapeInShapeSorter | var0 | 0/10 | 7/10 |
| PutKnifeInKnifeBlock | var0 | 0/10 | 10/10 |
| PickAndLift | var0 | 0/10 | 10/10 |
| StackBlocks | var0 | 0/10 | 6/10 |
| EmptyContainer | var0 | 0/10 | 9/10 |
| BlockPyramid | var0 | 0/10 | 8/10 |

---

## 四、遇到的问题与解决思路

### 4.1 Headless 模式下的物理引擎崩溃
- **问题**：部分任务（如 PlaceCups）在初始化时 CoppeliaSim 进程崩溃（ACCESS_VIOLATION）。
- **原因**：任务初始化中调用了 `SpawnBoundary` 进行随机物体放置，该操作依赖物理引擎碰撞检测，在 headless 模式下不稳定。
- **解决**：**按导师要求不再对任务初始化做任何 Monkey-patch**（不再替换 `SpawnBoundary`、不强制静态工作区），
  以保证随机初始化的真实性；改为在评测流水线层做**进程隔离**（每个 episode 使用全新环境实例），
  崩溃的 episode 如实计为失败。headless 下同时禁用视觉传感器、调优物理参数（dt=2ms / substeps=10）以降低崩溃率。

### 4.2 任务初始化阶段的 IK 校验失败
- **问题**：PressSwitch、PutKnifeInKnifeBlock 在 reset 阶段多次重试后失败，V-REP 返回错误码 -1。
- **原因**：场景初始化时会进行机械臂逆运动学（IK）可行性校验，headless 模式下 IK 求解不稳定。
- **解决**：该错误码与运动规划插件加载失败同源，随插件宿主库路径修复（见 4.4）而消失。
  按导师要求，现**不再对任何任务做"静态工作区 / 跳过校验"的补丁**（`_patch_task_for_headless()` 为空操作），
  所有任务使用真实随机初始化，初始化失败按真实失败计入。

### 4.3 长时序任务超时与 shutdown 段错误
- **问题**：PutRubbishInBin 等长时序任务在规划器执行超时后，看门狗杀死 sim 进程，随后 success() 访问已死亡的 PyRep 对象句柄触发 0xC0000005 段错误，导致结果 JSON 无法写入。
- **原因**：规划器超时时看门狗杀死 sim，但后续 success() 检查仍尝试访问已失效的 C++ 句柄。
- **解决**：
  1. 为长时序任务设置独立的规划超时预算，避免无谓的 480s 空等。
  2. 超时后**不修改任何成功判据**，如实记为失败。
  3. shutdown 流程中先杀死 sim 进程再调用 env.shutdown()，避免仍在运行的规划器线程访问死亡 sim。
  4. 规划器超时后跳过 reset_to_default_pose，防止与存活的规划器线程冲突。

### 4.4 运动规划插件未加载（R1）与传送式假成功（R2）
- **问题**：headless 下每个 episode 都出现
  `simExtOMPL: error: could not find or correctly load the CoppeliaSim library`
  （`simExtIK` / `simExtGeometric` / `simExtImage` / `simExtICP` 同理），
  RLBench 的位姿规划动作 100% 失败并抛 `The call failed on the V-REP side. Return value: -1`，
  整条运动规划链路被降级为裸 IK + 直接关节控制。
- **原因**：这些插件通过 `GetModuleFileNameA(NULL)` **自算宿主库路径**（主模块目录 + `coppeliaSim.dll`）。
  而 PyRep 是**进程内**启动 CoppeliaSim，进程主模块是 `python.exe`，其目录下没有该 DLL，故必然加载失败。
- **解决**：改用位于 CoppeliaSim 根目录内的 `python.exe` 启动评测脚本，使宿主模块目录正确。
  修复后 OMPL / IK 插件正常加载，机械臂报错由插件级错误变为 RLBench 原生的
  `A path could not be found`，`planner_success` 由 85.0% 升至 90.9%。
- **同时移除的假成功（R2）**：早期为提升通过率，代码中曾存在
  "把物体直接传送到成功接近传感器"或"把末端标记物传送到目标位姿即宣告成功"的兜底。
  这类操作会改写成功判据并污染运动学参考系，已按导师要求**全部删除**；
  `success()` 现为 RLBench 原生判据的**纯包装函数**，`_patch_task_for_headless()` 为空操作，
  所有任务均使用真实的随机初始化。

---

## 五、后续计划

### 短期（1-2 周）
1. **修复抓取执行环节**：非 joint 任务的当前主导瓶颈是抓取——夹爪在物体附近闭合却夹不住
   （原生判据 `NothingGrasped=True` / `GraspedCondition=False`）。计划修正后处理中
   "按压 / 滑动"启发式的误判、以及抓取 / 滑动路由词表不全的问题。
2. **规划末期收敛误差**：OMPL / IK 规划终点仍残留约 0.10 m 误差，结合闭环 IK 小步推进进一步收敛，
   减少回退到启发式兜底的次数。
3. **失效归因报告**：已按"感知 / 代码生成 / 规划执行"三阶段输出归因报告，并给出 R1/R2 修复前后对照。

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
- **核心修改文件**：`src/envs/rlbench_env.py`（headless 兼容性、物理层补偿、删除传送式假成功）、
  `src/interfaces.py`（价值图生成与动作路由）
- **评测入口**：`experiment_results/run_real10.py`（22 个任务变体 × 10 次，随机初始化 + 原生判据）
