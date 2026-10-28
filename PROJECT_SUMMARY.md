# 基于 VoxPoser 的机器人操作实验进展总结

> 汇报人：本人
> 更新时间：2026-10-24
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

| 测试项 | 基线（禁用物理补偿） | 启用物理补偿（严格口径，2026-10-23） |
|---|---|---|
| `planner_success` | 85.0%（187/220） | **87.3%（192/220）** |
| `env_success`（整体） | 8.2%（18/220） | **35.5%（78/220）** |
| joint / 开关类（PushButton / LampOff / PressSwitch，6 变体） | 13.3%（8/60） | **100%（60/60）** |

> **关于早期"100%"数字的更正**：早期汇总中曾出现"22/22 = 100%"，其一部分来自
> "把物体直接传送到成功接近传感器"或"把末端标记物传送到目标位姿即宣告成功"的人工兜底。
> 这类兜底会**改写成功判据**，属于假成功，已按导师要求**全部移除**。
> 上表为移除后、仅依赖真实物理执行与原生判据的**严格口径**结果。
> 同时修复了 CoppeliaSim 运动规划插件（OMPL / IK）的加载问题，`planner_success` 由 85.0% 升至 87.3%。

**分类结果（严格口径）：**

| 类别 | 变体数 | env_success | 说明 |
|---|---|---|---|
| joint / 开关类 | 6 | 100%（60/60） | 物理层补偿的主要受益者 |
| 抓取 / 堆叠 / 搬运类（非 joint） | 16 | 11.3%（18/160） | 受规划末期收敛误差与抓取执行环节限制 |

**非 joint 任务的进展（2026-10-23 基线）**：TakeUmbrellaOutOfUmbrellaStand 8/10、
ReachTarget 5/10、PutRubbishInBin 4/10、MeatOffGrill 1/10。
**SlideBlockToTarget 在 2026-10-24 定向修复后达到 8/10**（同一严格口径，10/10 planner），
**PickAndLift 在 2026-10-24 抓取链路修复后达到 9/10**（10/10 planner），
**PlaceCups 在 2026-10-24 放置链路修复后达到 5/10（10/10 planner），
2026-10-25 改为按辐条顶面定释放高度后先测得 9/10、同一代码重测为 3/10**（两批合计 12/20，
10/10 planner；杯子在单根辐条上一点支撑，落定过程敏感），
**StackBlocks 在 2026-10-25 多轮 pick-place 修复后达到 8/10**（10/10 planner），
**MeatOffGrill 在 2026-10-25 目标解析 + 释放高度修复后达到 9/10**（9/10 planner），
**TakeLidOffSaucepan 在 2026-10-25"举升进传感器"修复后达到 10/10**（10/10 planner），
这些结果尚未并入全量基线。

**逐任务 env_success（10 次/变体，2026-10-23 基线）：**

| 任务 | 变体 | env_success |
|---|---|---|
| PushButton | var0 / var1 | 10/10 · 10/10 |
| LampOff | var0 / var1 | 10/10 · 10/10 |
| PressSwitch | var0 / var1 | 10/10 · 10/10 |
| TakeUmbrellaOutOfUmbrellaStand | var0 | 8/10 |
| ReachTarget | var0 | 5/10 |
| PutRubbishInBin | var0 | 4/10 |
| MeatOffGrill | var0 | 1/10 基线 → **9/10**（10-25 修复后） |
| SlideBlockToTarget | var0 | 0/10 基线 → **8/10**（10-24 修复后） |
| StackCups | var0 / var1 / var2 | 0/10 · 0/10 · 0/10 |
| TakeLidOffSaucepan | var0 | 0/10 基线 → **10/10**（10-25 修复后） |
| PlaceCups | var0 | 0/10 基线 → 5/10（10-24）→ 9/10 后**重测 3/10**（10-25 同一代码，合计 12/20） |
| PlaceShapeInShapeSorter | var0 | 0/10 |
| PutKnifeInKnifeBlock | var0 | 0/10 |
| PickAndLift | var0 | 0/10 基线 → **9/10**（10-24 修复后） |
| StackBlocks | var0 | 0/10 基线 → **8/10**（10-25 修复后） |
| EmptyContainer | var0 | 0/10 |
| BlockPyramid | var0 | 0/10 |

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
  `A path could not be found`，`planner_success` 由 85.0% 升至 87.3%。
- **同时移除的假成功（R2）**：早期为提升通过率，代码中曾存在
  "把物体直接传送到成功接近传感器"或"把末端标记物传送到目标位姿即宣告成功"的兜底。
  这类操作会改写成功判据并污染运动学参考系，已按导师要求**全部删除**；
  `success()` 现为 RLBench 原生判据的**纯包装函数**，`_patch_task_for_headless()` 为空操作，
  所有任务均使用真实的随机初始化。

### 4.5 SlideBlockToTarget 的三个执行层根因（2026-10-24）

规划器 10/10 但环境 0/10，用两个只读探针逐项排除后定位到三个独立缺陷：

1. **原生条件元组误读**：`Condition.condition_met()` 返回 `(met, done)`，`done` 恒为 `False`。
   直接 `bool(tuple)` 恒真；改用 `all(tuple)` 后**恒假**——方块距传感器 32mm 时
   `env.success()=True`，控制端却一直读到 `conds=[False]`，继续把方块顶出判定区。
   修复：只取第 0 个元素。
2. **判定区是圆环而非点**：`success` 传感器与场景 `target` 完全重合；方块中心距传感器
   **22~38mm** 才检出（yaw 0~45°、各方向一致），22mm 内是死区（方块 AABB 包住传感器）。
   修复：推块目标改为"把方块送进环带"，预停线 36mm，并增加死区逃逸推送。
3. **推块过冲与滑行**：终点按"方块中心 + d·L"计算，而 EE 起点本就在方块后方 5.5cm，
   等于多走 5.5cm；低摩擦下方块滑行速度可达 EE 的 3 倍（EE 126mm vs 方块 400mm），
   一步跨过 16mm 宽的环带导致 `stop_when` 漏检。
   修复：终点改为相对 EE 起点；推速降到 1.5mm/步，`horizontal_push_continuous()`
   新增 `settle_steps=6` 步间稳定；远端方块（>250mm）先经工作空间上方中转再逼近。

**验证**：同一严格口径下 10 连跑 **8/10**（10/10 planner）。剩余 2 次失败的方块距传感器
315mm / 286mm，背面在机械臂可达包络之外（逼近残差 164mm / 186mm），属工作空间限制。

### 4.6 PickAndLift 的四个抓取链路根因（2026-10-24）

基线 0/10（planner 9/10），失败签名是"夹爪握着物体却从不举起"或"目标解析到干扰块"。
用只读探针 `probe_pick_lift.py` 逐项实测后定位到四个独立缺陷：

1. **语义名解析命中干扰块**：场景同时有 `pick_and_lift_target`（handle 84）与
   `stack_blocks_distractor0/1`。`name2ids["block"]` 正确指向 handle 84，但
   `get_object_pos_by_name("block")` 走的是场景名包含匹配，返回干扰块
   `[0.354, 0.131]`，而目标块实际在 `[0.047, −0.131]`。
   修复：优先走 `name2ids`；同名多句柄时按世界 AABB 体积取实体。
2. **成功传感器藏在 ConditionSet 内层**：原生成功条件是
   `GraspedCondition(目标) ∧ DetectedCondition(目标, 上方传感器)`，只看顶层 `_detector`
   得到 `None`，放置目标退回到不可达的 affordance（z≈1.5）。
   修复：递归进入 `ConditionSet._conditions` 查找传感器。
3. **误当"抓取+放置+释放"任务**：成功条件含 `GraspedCondition`，一旦释放即永久失败。
   修复：新增 lift-only 分支——保持夹持把物体举进传感器
   （目标 z = 传感器 z + 持物挂高约 22mm，按 ±2/4/6cm 扫高度，每步读原生 `env.success()`），
   已握持时同样执行该分支。
4. **后续 waypoint 破坏已满足的成功**：举升成功后剩余 waypoint 继续驱动机械臂，
   把目标移出传感器，终态又变回 `False`。
   修复：waypoint 循环入口检查原生 `env.success()`，已满足即停止剩余 waypoint（只读，不改判据）。

**验证**：同一严格口径下 10 连跑（r19–r28）**9/10**（10/10 planner）。
唯一失败 r24 的目标块在 x=−0.191，逼近残差 119mm，属该次随机初始化的可达性限制。

### 4.7 PlaceCups 的放置目标与"挂杯"难题（2026-10-24）

基线 0/10（planner 9/10）。旧日志的失败链是 `place target via affordance target`
落点几乎就是杯子自身位置，释放后 `NothingGrasped met=True` 但 `OrConditions met=False`。
三个只读探针定位到：

1. **成功传感器藏在 OrConditions 内层**：成功条件是
   `NothingGrasped ∧ OrConditions[DetectedCondition(mug0, success_detector0/1/2)]`，
   三个传感器装在三根辐条上（z=0.9618/0.9009/0.9618）。只看顶层 `_detector` 得到 `None`。
   修复（与 PickAndLift 共用）：递归进入 `ConditionSet._conditions`。
2. **传感器是向上开口的锥体**：杯心低于传感器平面时完全不检出；z ∈ [sensor, sensor+0.10]、
   顶面半径 ≥8cm 时各方向都检出。因此成功形态是"杯子停在辐条上（杯心 z≈1.015）"。
3. **被动松手不稳定**：在架子周围做 60 点自由落体投放，只有 3 个点留在架子上并判成功（5%）。
   当前代码在传感器 XY 上方 18cm 释放：杯子落在辐条上则成功，弹落到台面
   （z≈0.79，距传感器 170~450mm）则失败。

**释放高度对照（定向样本）：** 高释放（EE≈1.19）5/10；低释放（EE≈1.05）0/3——
夹爪指板下探到 z≈1.01 会扫到辐条/架柱把杯子拨飞；沿辐条外移 3cm + 辐条高度释放 0/2——
杯子从辐条滑落。最终**保留高释放，不作 PlaceCups 专用改动**（实验改动已回退）。

**验证**：同一严格口径下 10 连跑 **5/10**（10/10 planner），对照基线 0/10。
剩余失败根因是"手柄朝向辐条"缺控制：杯子随机偏航，只有手柄恰好套住辐条才会留在架上。
下一步计划用 AABB 与视觉网格中心差估计手柄方向，在持物状态下绕 z 旋转对齐后再小幅下放。
（2026-10-25 在改动后的构建上重测 10 次：3/10，放置参数完全一致——高释放 EE≈1.19、
`rel_z≈-42mm`；成功样本仍停在辐条上（z≈1.01，d≈51mm），差异来自手柄偏航的随机性，
不是回归。）

### 4.8 StackBlocks 的多物体放置（2026-10-25）

**问题**：基线 0/10（planner 10/10）。成功条件是
`DetectedSeveralCondition(target_blocks, stack_blocks_success, 2) ∧ NothingGrasped`，
而单轮 Fix2 只搬一只方块。

**只读探针（`probe_stack_blocks.py`）**：`stack_blocks_success` 的检出体是**圆柱**——
半径≈30mm，z 从台面（0.752）延伸到 +120mm。两只 50mm 方块的**中心**不可能同时落在
30mm 圆内并排摆放，所以必须**叠放**。

**四处修复**：

1. **支撑面门控**：`_base_top = place_z + 0.14` 是 StackCups 专用经验值（其传感器在底座杯
   底部）；误用到 stack_blocks 会让方块从 14cm 高空释放、弹飞 0.84m。多物体任务改用
   放置点高度作支撑面，支撑面扫描只接受距放置点 8cm 内的同类物体。
2. **释放前横向居中**：descend 的 50mm 容差允许 32mm 横向残差，已超出 30mm 检出圆。
   现在在搬运高度按"物体 AABB 中心 → 放置点"的实测偏差平移 EE（纯 IK，无传送），
   下放保持纯垂直。
3. **补轮 `_multi_round_pick_place()`**：复用抓取/搬运/下放/释放原语，把成功条件里
   其余物体逐个送到圆内（每只叠在圆内已有的那只顶面上）；每轮读原生 `env.success()`
   决定是否收工。同时覆盖两种失败模式：首轮抓取失败、以及计划层一直握着方块不松手
   （此时先把手上那只放下）。
4. **成功锁存**：一旦原生 `env.success()` 为 True，`execute()` 在本次 episode 内只读返回。
   RLBench 的成功即终止，而跑批会继续执行计划里剩余的 composer 步骤——实测有一次
   达成成功后又把传感器处的方块抓走，终态回落到 False。

**验证（严格口径，随机初始化，每 episode 全新 env）**：

| 样本 | env_success | planner |
|---|---|---|
| StackBlocks var0 ×10（主样本） | **8/10** | 10/10 |
| 追加 ×5（接近/下放增加一次 OMPL 回退重试） | 4/5 | 5/5 |

两次失败分别是：抓取假成功（`cnt=2`，跟随复核漂移 199mm）与接近 IK 卡死（残差 381mm）。
追加样本中一次失败是两只方块停在 42/44mm 处（刚好在 30mm 圆外），已把"放置点附近
不再搬"的阈值从 6cm 降到 3cm，使这类"近但没进圈"的方块可再对正。

### 4.9 插入类任务：PutKnifeInKnifeBlock 的物理不可解判定（2026-10-25）

方案第 2 项是"槽位/朝向对齐原语"。先用 5 个只读探针把几何量清楚：

| 探针 | 结论 |
|---|---|
| `probe_knife_pose.py` | 刀设为静态时，中心落在传感器 xy±40mm、z∈[sensor, sensor+0.09] 即被原生检出，**与 yaw 无关** |
| `probe_knife_free.py` | 静态"检出"位姿很多，但都是把刀直接摆进刀座内部（穿透），非物理可达 |
| `probe_knife_scene.py` | 场景只有一把刀（h=91）+ 刀座（h=84）+ 刀座显示体（h=88）；**刀座本身就被传感器检出** |
| `probe_knife_solid.py` | 从顶面上方竖直/水平释放：竖直落定 `bottom=0.912`=顶面、水平 `0.921`，**0 检出** |
| `probe_knife_openings.py` | 顶面 1cm 栅格 25 点 + 45° 斜放 5 点：落点只有"顶面 0.912"或"台面 0.748"，**0 检出** |

**判定**：刀座碰撞体 4.1×8.0×15.6cm、顶面 z=0.911；`success` 传感器在刀座**底部**
（z=0.766），实测检出带 z∈[0.766, 0.866] **整段位于实心碰撞体内部**。25+5 个自由释放点
无一能进入 ⇒ 在"不穿透实体"前提下该任务在本仿真资产下**物理不可解**（与 StackCups 同类
的任务级限制），不是规划器/抓取缺陷。

**代码侧**：

1. 实现插入原语 `_insert_knife_into_block()`：抬到旋转安全高度 → 长轴转到世界 -Z
   （刃朝下，纯 IK，失败时 OMPL 重试）→ 按**刀句柄** AABB 对齐传感器 XY → 沿刀座两条
   水平轴枚举 yaw 并 1cm 步长下探，全程读原生 `is_detected`。
2. 刀的所有几何量改按 **PyRep 句柄**读取：`get_object_pos_by_name('knife')` 会被 VoxPoser
   语义别名（knife1 / knife3 / black knife）解析到别的物体（实测先报 7mm、后又报 99mm）。
3. 下探"停滞一次即收手"且 AABB 变 NaN 立即退出：硬压刀座实测会把刀矩阵压成 `nan`，
   之后所有旋转 `err=inf`。
4. 增加**实心碰撞体可行性守卫**：传感器 XY 在刀座足迹内且检出带上沿低于刀座顶面时，
   直接 SKIP 插入原语并打印原因，交回通用路径如实失败（避免用穿透伪造成功）。

**现场**（严格口径，planner 4/4 成功）：竖直下放 0/1、yaw 搜索 0/1（读数被别名污染）、
句柄测量 0/1（硬压导致 NaN）、加守卫 0/1（`slot-insert SKIP` 干净触发、无污染）。
任务保持 0/10，归因为**资产级物理限制**；要提升只能替换刀座资产或放宽任务判据。

### 4.10 PlaceCups：限制因素其实是释放高度，不是手柄朝向（2026-10-25）

10-24 的放置目标修复后 PlaceCups 为 5/10，后续重测 3/10，当时的假设是"只有杯柄恰好套住
辐条才留在架上"。只读探针 `probe_place_cups_handle2.py` 直接检验了这个假设：

- 三根 `place_cups_holder_spoke{0,1,2}` 长 95mm、粗 13mm、**上仰 20°**，
  `success_detector{i}` 与辐条 i 的中心**完全重合**（d=0mm）；
- 杯子局部 AABB 为 65.6 × 97.3 × 83.7mm，手柄沿局部 **±Y**（网格质心在 Y 上偏 5mm）；
- 在同一投放点，手柄**同向 / 垂直 / 45°** 三种朝向落下后**都留在架上且被原生检出**；
  真正的差异是**释放高度**：离传感器 6cm 落下 → 杯心 z≈1.02（在判定锥内）；
  12/18cm 落下 → 杯心 z≈1.07 / 1.13（在判定锥上沿 sensor+0.10 之上，全部 False）。

⇒ 旧的 `_base_top = place_z + 0.14` 经验值让 EE 在 ≈1.19 释放（杯底离辐条顶约 7cm），
杯子多数次停在架顶或弹到台面——这才是 3/10 的来源。现改为按辐条自身反算：
`rod_top = sensor_z + 6.6mm`，目标杯心 = `rod_top + 杯半高 + 2mm`，EE = 目标杯心 − 实测
"夹爪→杯心"偏移；杯子只下落 1~2cm 即进入判定带。**不做手柄 yaw 对齐**（探针已证不影响检出）。

**验证**（严格口径，随机初始化，每 episode 全新 env）：10 连跑 **9/10**（planner 10/10），
对照此前 3/10。唯一失败 r6 属抓取侧：`grasp_with_retry` 连续两次 `cnt=0`，三个杯子都留在
台面（距传感器 451~514 mm），与释放高度无关。

**复现性更正（2026-10-25，同一代码）：9/10 不可复现，重测为 3/10。** 第二批 10 连跑
（run10c）planner 仍是 10/10、env 只有 3/10；两批的释放参数完全一致
（EE≈1.05、`rel_z≈−42 mm`、横向居中残差 0~3 mm），唯一的代码差异是更严格的抓取确认
（在本任务上从未触发），因此差异属**任务级方差**：两批合计 **12/20**。
样本呈清晰的双峰——成功的杯子都停在传感器正上方（杯心 z≈1.01~1.02，XY 偏差几毫米），
失败的杯子都滑落到台面（z≈0.79，水平偏离 89~261 mm）。杯子实际上是**单根辐条上的一点支撑**，
能否站稳取决于随机落定过程。

追加对照实验：把放置点改为**三根辐条的环心**（理论上让杯子落在环上而非单根辐条）。
结果 **1/10**——三根辐条 XY 的均值正是杯架的**中央立柱**，杯底直接压到立柱顶面，
10 次里 9 次 `descend-to-place` 卡在 97~99 mm 无法下放（唯一成功还是杯子侧滑到辐条上）。
该改动已回退，放置点保持“单根辐条中心”。

### 4.11 TakeLidOffSaucepan：0/10 → 10/10，锅盖必须抬升 16~22 cm（2026-10-25）

只读探针 `probe_lid_sensor.py` 实测几何：

| 量 | 结果 |
|---|---|
| 锅盖 grasp point 初始 | z=0.853 |
| success 传感器 | z=0.968，位于锅盖**正上方**（XY 偏移 0） |
| 检出窗口（lid z） | **[1.013, 1.073]** → 必须抬升 **16~22 cm** |
| 横向容差 | 偏移 ≤8 cm 仍检出，12 cm False |

基线的失败**不在抓取**：每 episode 都成功夹住锅盖（`cnt=1`）。计划把锅盖抬到"锅上方 20cm"
（≈窗口下沿），随后执行 `open gripper` 释放——而成功条件是
`ConditionSet([GraspedCondition(lid), DetectedCondition(lid, success)])`，一旦释放就永久失败。

修复：`take_lid_off_saucepan` 在已握持时调用 `_lift_to_sensor('take-lid')`，
沿传感器 ±2/4/6 cm 逐步举升，每步读原生 `env.success()`；同时抑制 press-mode 与释放
（`press-mode suppressed (lid task is lift-only while holding)` / `skip release`）。
**验证：10 连跑 10/10**（planner 10/10）。

### 4.12 MeatOffGrill：1/10 → 9/10，三个独立缺陷（2026-10-25）

1. **泛称 `meat` 命中不可抓的视觉网格**：每 episode 的 `Post-action` 都把 `meat` 解析到
   `"silver meat"`（top_z≈1.09~1.13，网格不可抓），夹爪在网格上空闭合；基线 6/10
   出现三次重试 + 25 点扫描全空（`detected=[]`）。
   修复：`meat` 时按指令点名解析 `chicken` / `steak`。
2. **叠放支撑面把悬空网格当支撑**：`_support_top=1.317` → 目标物体中心 1.341、
   EE 在 **1.349** 释放（离传感器 29 cm），肉弹飞 411 mm。
   修复：肉任务跳过叠放逻辑，按 `sensor_z + 半高 + 2cm` 释放。
3. **计划自己先抓住肉时 Fix2 整段被跳过**：Fix2 的抓取→放置分支要求 `grasped_before==0`，
   而 composer 的 affordance 动作常在航点处就闭爪（`grasped_before=1`），后续只剩
   `move/open` → 肉一直握在手里。新增原语 `_place_at_sensor()`：已握持时把物体送到
   传感器上方、按**物体**（而非 EE）居中，再"闭爪抬 1cm → 张开"。
   本轮 9 次成功全部走这条路径。

**验证：10 连跑 9/10**（planner 也可以正常收尾）。当时唯一失败 r7 被记为"IK 侧发散"，
后续追踪纠正了归因：NaN 更早由 **PyRep 自己的路径执行**注入——
`ArmConfigurationPath._step_motion` 在相邻两个路径构型重合时
`t=(pos-len_i)/(len_{i+1}-len_i)` 出现 0/0，`qs=p1+dx*t` 变成 NaN，并被直接写入动态关节
（绕过 `_set_joints_smooth`）。NaN 一旦进入动力学求解器就无法回滚，`_ik_target_solve`
只是第一个**读到** NaN 的地方（给 IK 目标 Dummy 设位姿加 step 并不会驱动机械臂）。

**P0 修复（2026-10-25）**：在 `RobotComponent.set_joint_target_positions` 上挂有限性闸门，
非有限目标整条丢弃（伺服保持上一帧有限目标），并在 `_set_joints_smooth` 补 `cur` 有限性校验。
修复后 10 集重跑：**9/10 env、10/10 planner、0 次 NaN-poisoned、0 次 900s 超时**；
闸门在 9/10 集里累计拦截 25 次 0/0 注入（正是修复前毒死 r7 的同一事件）。
剩余失败 r1 为真实抓取落空（`direct-close detected=[]` ×3、扫描全空），与 IK 无关。

**P1 负结果（2026-10-26，已回退）**：尝试把每个 2cm 子步的朝向与位置一起 slerp，
并在关节跳变被拒时把步长折半重试。10 集重跑 env 8/10，ik-step 平均推进 0.115 → 0.045 m、
关节跳变拒绝 14 → 223 次；机制指标与结果同时变差，已完整回退到 `4900acc`。

**抓取重试重定位（2026-10-27）**：run6 的 r1 是"接近阶段失败、EE 停在物体上方 20~27cm，
却只原地再下压 15mm"——永远够不到。改为空抓后按"物体 XY + top_z−15mm"闭环重定位
（仅当 `target_pos` / `top_z` 可用；否则退回原下压逻辑）。
run8：**9/10 env、10/10 planner、0 poisoned、0 超时**；r9 由重试直接救回
（首次空抓 → 重定位 err 0.5mm → 第二次 `detected=['chicken']`）。
跨任务回归 PlaceCups×3 + PickAndLift×2：**5/5 env、5/5 planner**，重试 0 次触发，无副作用。

---

## 五、后续计划

### 短期（1-2 周）
1. **修复抓取执行环节**：非 joint 任务的当前主导瓶颈是抓取——夹爪在物体附近闭合却夹不住
   （原生判据 `NothingGrasped=True` / `GraspedCondition=False`）。计划修正后处理中
   "按压 / 滑动"启发式的误判、以及抓取 / 滑动路由词表不全的问题。
   （进展：PickAndLift 的抓取链路已修复至 9/10；下一步对 PlaceCups 做同类只读探针归因。）
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
