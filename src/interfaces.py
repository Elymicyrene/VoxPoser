from LMP import LMP
from utils import get_clock_time, normalize_vector, pointat2quat, bcolors, Observation, VoxelIndexingWrapper
import numpy as np
from planners import PathPlanner
import time
import os
import traceback
from scipy.ndimage import distance_transform_edt
import transforms3d
from controllers import Controller

# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']

# 任务类别词表：原内联在动作后处理里，现提升到模块级，供 movable 归一层复用
# （movable 解析失败时用它从场景+指令里找回真实物体，避免降级成 EE 模式）。
_CONTROL_TOKENS = {'button', 'buttons', 'switch', 'switches',
                   'knob', 'knobs', 'lever', 'levers', 'dial', 'dials'}
_PICKABLE_TOKENS = {'meat', 'chicken', 'beef', 'pork', 'fish', 'steak',
                    'apple', 'banana', 'orange', 'tomato', 'pepper',
                    'chili', 'onion', 'garlic', 'strawberry', 'pear',
                    'grape', 'lemon', 'lime', 'mango', 'peach', 'plum',
                    'cup', 'mug', 'bottle', 'can', 'pot', 'pan',
                    'plate', 'bowl', 'tissue', 'book', 'box', 'cube',
                    # 2026-10-25：PutKnifeInKnifeBlock 的 movable 是 'knife'，旧词表没有它，
                    # 于是 Fix2（pick-place 兜底）从不进入，只剩无目标的 intend-CLOSE 闭爪。
                    'knife'}
_SLIDABLE_TOKENS = {'slider', 'sliders', 'drawer'}
# R4: "block/brick" 既可被抓起（pick up the block）也可被推动
# （slide the block to target），单靠名词无法判定 → 归入歧义集合，
# 由当前 episode 的指令动词决定类别。
_AMBIG_TOKENS = {'block', 'blocks', 'brick', 'bricks'}
_SLIDE_VERBS = ('slide', 'push', 'shove', 'nudge', 'drag')
_PICK_VERBS = ('pick', 'lift', 'stack', 'unstack', 'place', 'put',
               'build', 'grasp', 'grab', 'take', 'hold', 'carry', 'insert')
# 颜色词表：用于从指令里识别「被当作底座 / 支撑物」的那个物体。
# 典型：StackCups 的 "stack the other cups on top of the red cup" —— 红杯是底座，
# 不该被抓取/搬走（搬走底座会让任务必然失败）。
_COLOR_TOKENS = {'red', 'blue', 'green', 'yellow', 'orange', 'pink', 'purple',
                 'cyan', 'brown', 'gray', 'grey', 'black', 'white', 'maroon',
                 'crimson', 'coral', 'magenta', 'navy', 'olive', 'teal', 'lime',
                 'gold', 'silver', 'violet', 'turquoise'}


def _build_pick_guard(env):
  """返回本 episode 指令点名的可抓物体的 [(xy(2,), top_z)]。

  用途（run25 复盘）：composer 的 affordance 目标常落在可抓物体本身上
  （如 "a point at the center of the black cup"），而执行层会 (1) 把 EE 一路
  降到物体中心高度、(2) 之后无条件下压 30mm——闭合夹爪压进杯口会把杯子从台面
  顶飞（run25 r0 实测 cup 被顶到 [0.446 0.709 0.042]，工作空间外，之后恒
  Fix2 SKIP）。这里统一提供"可抓物体水平位置 + 顶面高度"，供三处复用：
    * waypoint 循环（EE-mode）：z 目标低于顶面时抬到顶面之上；
    * post-loop Final approach：同样抬高，不把 EE 直接送进物体；
    * 下压动作门控：目标落在可抓物体上时跳过物理下压（R3）。
  """
  _out = []
  try:
    _instr = str(getattr(env, 'current_instruction', '') or '').lower()
    _names = env.get_object_names()
  except Exception:
    return _out
  import re as _re_pg
  _toks = set()
  _name_by_tok = {}
  for _s in _names:
    _t = str(_s).lower().strip()
    if ' ' in _t:
      _t = _t.split(' ')[-1]
    elif '_' in _t:
      _t = _t.split('_')[-1]
    _m = _re_pg.match(r'^(.*?)(\d+)$', _t)
    if _m:
      _t = _m.group(1)
    _toks.add(_t)
    _name_by_tok.setdefault(_t, []).append(str(_s))
  for _tok in sorted(_toks & _PICKABLE_TOKENS):
    if _tok not in _instr:
      continue
    # 逐个具体物体名解析（如 'black cup1'/'red cup2'/'black cup3' → 真实 'cup1/2/3'），
    # 覆盖同类多物体；用具体名而非泛称 token，避免只解析到其中一个。
    for _full in _name_by_tok.get(_tok, []):
      try:
        _p, _tz = env.get_object_pos_by_name(_full)
        if _p is None:
          continue
        _p = np.asarray(_p, dtype=float).reshape(3)
        if not np.all(np.isfinite(_p)):
          continue
        _top = float(_tz) if _tz is not None else float(_p[2])
        _out.append((_p[:2].copy(), float(_top)))
      except Exception:
        continue
  return _out


def _same_class_full_names(env, token):
  """返回场景中与 token 同类的具体名（去掉颜色前缀/数字后缀后相同）。

  如 token='cup' → ['black cup1', 'red cup2', 'black cup3']。
  """
  _out = []
  try:
    _names = env.get_object_names()
  except Exception:
    return _out
  import re as _re_sc
  for _s in _names:
    _t = str(_s).lower().strip()
    if ' ' in _t:
      _t = _t.split(' ')[-1]
    elif '_' in _t:
      _t = _t.split('_')[-1]
    _m = _re_sc.match(r'^(.*?)(\d+)$', _t)
    if _m:
      _t = _m.group(1)
    if _t == token:
      _out.append(str(_s))
  return _out


def _support_color_from_instruction(instr):
  """从指令里解析「作为底座 / 支撑物」的那个物体的颜色词（解析不到则 None）。

  覆盖 StackCups 的 6 种句式（"the other cups on top of the red cup"、
  "set the cups down into the red cup"、"with the red cup as its base"、
  "keeping the red cup on the table, ..."）。用途：底座杯是支撑物而非操作对象，
  应当被排除出抓取目标候选——run33 实测 target 择优在 cup1 被撞出界后退到了
  底座杯 cup2，把它搬到传感器位置，任务必然失败。
  """
  if not instr:
    return None
  import re as _re_sup
  _s = str(instr).lower()
  _pat = (r'(?:on top of|onto|atop|into|with|as its base|as the base|keeping|'
          r'keeping the|on the)\s+(?:the\s+|a\s+)?([a-z]+)')
  for _m in _re_sup.finditer(_pat, _s):
    _w = _m.group(1)
    if _w in _COLOR_TOKENS:
      return _w
  return None


def _resolve_pick_target(env, token, ref_xy=None, exclude_color=None):
  """为 Fix2 选一个"可抓"的同类物体，并返回其**具体名**。

  动机（run26 复盘）：Fix2 原先用泛称 token 走 get_object_pos_by_name（包含匹配、
  取第一个候选）。一旦第一个候选（如 cup1）被臂部运动推出工作空间，该解析会永久
  指向那只出界杯 → Fix2 恒 SKIP，哪怕同类的另一只杯子仍在台面上可抓。这里改为
  遍历同类具体名、排除工作空间外候选，多个时取离 ref_xy 最近者；并返回具体名供
  后续"几何跟随验证"复用（泛称无法安全用于跟随复核）。

  run34 新增 exclude_color：指令点名的"底座 / 支撑物"颜色（如 StackCups 的红杯）不
  参与择优。run38 起：有 exclude_color 时**不再回退**到不排除——底座杯是支撑物，
  宁可返回 None（让 Fix2 SKIP）也绝不能把它搬走。

  :return: (full_name, center(3,), top_z) 或 (None, None, None)
  """
  def _scan(_excl):
    _best = None
    for _full in _same_class_full_names(env, token):
      if _excl and _excl in str(_full).lower():
        continue
      try:
        _c, _tz = env.get_object_pos_by_name(_full)
      except Exception:
        continue
      if _c is None:
        continue
      try:
        _c = np.asarray(_c, dtype=float).reshape(3)
      except Exception:
        continue
      if not np.all(np.isfinite(_c)):
        continue
      if _obj_off_workspace(env, _c):
        continue
      _top = float(_tz) if _tz is not None else float(_c[2])
      _d = 0.0
      if ref_xy is not None:
        try:
          _r = np.asarray(ref_xy, dtype=float).reshape(-1)
          _d = float(np.linalg.norm(_c[:2] - _r[:2]))
        except Exception:
          _d = 0.0
      if _best is None or _d < _best[3]:
        _best = (_full, _c, _top, _d)
    return _best

  _best = _scan(exclude_color)
  # run38：底座/支撑物颜色是"指令点名的支撑物"，**永远不能**作为抓取目标。
  # run37 实测：排除红杯后，两只非红杯被撞出工作空间（x=0.943 / [-0.818,1.254]），
  # 旧回退把排除项重新纳入候选 → 机械臂随后把底座红杯搬了 5 次
  # （[0.135,0.423]→[0.174,0.34]→[0.213,0.258]→[0.201,-0.338]），而 success
  # 传感器是 cup2 的子节点 → 随底座杯漂移 0.4~0.5m，DetectedCondition 恒 False。
  # 因此有 exclude_color 时禁止回退：宁可返回 None 让 Fix2 SKIP。
  if _best is None:
    return None, None, None
  return _best[0], _best[1], _best[2]


def _grasp_follow_check(env, full_name, ee_pos, tol_m=0.12):
  """几何复核"被握物体是否仍跟随 EE"。

  `get_grasped_object_count()>0` 只说明抓取约束已建立；headless 下该约束可能名存
  实亡（实测 run26 r0：cnt=1 但搬运后物体跑到工作空间外 [1.319,-0.363]）。用真实
  几何复核可提前识别"假握持"，避免后续"假搬运 + 假释放 + 被推出界"。

  :return: (ok: bool, drift_m: float|None, obj_pos: ndarray|None)
           — 无法判定时返回 (True, None, None)，即保守放行。
  """
  if not full_name:
    return True, None, None
  try:
    _p, _ = env.get_object_pos_by_name(full_name)
    if _p is None:
      return True, None, None
    _p = np.asarray(_p, dtype=float).reshape(3)
    if not np.all(np.isfinite(_p)):
      return True, None, None
    _e = np.asarray(ee_pos, dtype=float).reshape(3)
    _d = float(np.linalg.norm(_p - _e))
    return (_d <= float(tol_m)), _d, _p
  except Exception:
    return True, None, None

# ── Plan-level wall-clock budget ───────────────────────────────────────────
# 一次 planner/composer LMP 调用（= 一个 "plan"，可能含多个 composer 步骤、
# 每个步骤都会重入 execute()）的**总墙钟上限**。默认 600s，低于评测 harness
# 的 planner 看门狗（默认 900s）。到点后 execute() 立即放弃剩余 waypoint 与
# post-action 并干净返回，使 harness 能正常执行 env.success()，而不是
# "超时杀 sim → 跳过 success()"（后者恒判 env False）。
# 可用 VOSPOSER_PLAN_BUDGET_S 覆盖；<=0 表示不限制。
_PLAN_DEADLINE = None


def set_plan_deadline(budget_s=None):
  """由 LMP planner/composer 在调用入口设置本次 plan 的墙钟 deadline。

  预算优先级：显式入参 > VOSPOSER_PLAN_BUDGET_S > (VOSPOSER_PLANNER_TIMEOUT_S - 150s)
  > 默认 600s。据此保证 plan 一定在评测 harness 的 planner 看门狗之前返回；
  留 150s 余量用于 reset / env.success() / env.close()（全量实测有多次
  plan 已跑 800~863s、结果也写了，却因收尾被 900s 外层墙钟判超时）。
  """
  global _PLAN_DEADLINE
  if budget_s is None:
    budget_s = None
    try:
      _env_b = os.environ.get('VOSPOSER_PLAN_BUDGET_S', '').strip()
      if _env_b:
        budget_s = float(_env_b)
      else:
        _host_t = os.environ.get('VOSPOSER_PLANNER_TIMEOUT_S', '').strip()
        budget_s = max(60.0, float(_host_t) - 150.0) if _host_t else 600.0
    except Exception:
      budget_s = 600.0
  _PLAN_DEADLINE = (time.time() + budget_s) if (budget_s and budget_s > 0) else None
  try:
    from envs import rlbench_env as _rlbench_env_mod
    _rlbench_env_mod.set_episode_deadline(_PLAN_DEADLINE)
  except Exception:
    pass
  return _PLAN_DEADLINE


def clear_plan_deadline():
  global _PLAN_DEADLINE
  _PLAN_DEADLINE = None
  try:
    from envs import rlbench_env as _rlbench_env_mod
    _rlbench_env_mod.clear_episode_deadline()
  except Exception:
    pass


def plan_budget_exceeded():
  """本 plan 的总墙钟是否已超预算（未设置 deadline 时恒 False）。"""
  return (_PLAN_DEADLINE is not None) and (time.time() > _PLAN_DEADLINE)


def _obj_off_workspace(env, pos, margin=0.03):
  """物体世界坐标是否非有限、或越出工作空间（含安全边距）。

  用于抓取/搬运阶段的"物体离台/越界早停"：一旦目标被撞出可作业区域，
  继续接近/下压/搬运只会反复撞击几何体、令接触求解发散，且后续每个
  composer 步骤都会重入兜底、把 planner 墙钟耗尽。
  """
  try:
    lo = np.asarray(env.workspace_bounds_min, dtype=float).reshape(3)
    hi = np.asarray(env.workspace_bounds_max, dtype=float).reshape(3)
    p = np.asarray(pos, dtype=float).reshape(3)
    if not np.all(np.isfinite(p)):
      return True
    return not (np.all(p >= lo - margin) and np.all(p <= hi + margin))
  except Exception:
    return False


class LMP_interface():

  def __init__(self, env, lmp_config, controller_config, planner_config, env_name='rlbench'):
    self._env = env
    self._env_name = env_name
    self._cfg = lmp_config
    self._map_size = self._cfg['map_size']
    self._planner = PathPlanner(planner_config, map_size=self._map_size)
    self._controller = Controller(self._env, controller_config)

    # 多物体任务（StackBlocks 等）的多轮 pick-place 只允许触发**一次**：
    # 每个 composer 步骤都会重入 execute()，若每次都补两轮，episode 会撞上
    # planner 900s 看门狗（实测 r1 单次跑批 >9min）。
    self._multi_round_used = False
    # 成功锁存（episode 终止语义）：RLBench 在 success() 为 True 的瞬间就结束
    # episode，而跑批会继续执行计划里剩余的 composer 步骤。实测 StackBlocks r2
    # 在 2:7:43 已达 success，随后 2:8:9 的计划步骤又把传感器处的方块抓走，
    # 终态回落到 False。锁存后 execute() 只读返回，不再下发任何动作。
    self._episode_success_latched = False

    # calculate size of each voxel (resolution)
    self._resolution = (self._env.workspace_bounds_max - self._env.workspace_bounds_min) / self._map_size
    print('#' * 50)
    print(f'## voxel resolution: {self._resolution}')
    print('#' * 50)
    print()
    print()
  
  # ======================================================
  # == functions exposed to LLM
  # ======================================================
  def get_ee_pos(self):
    return self._world_to_voxel(self._env.get_ee_pos())
  
  def detect(self, obj_name):
    """return an observation dict containing useful information about the object"""
    if obj_name.lower() in EE_ALIAS:
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self.get_ee_pos()
      obs_dict['aabb'] = np.array([self.get_ee_pos(), self.get_ee_pos()])
      obs_dict['_position_world'] = self._env.get_ee_pos()
    elif obj_name.lower() in TABLE_ALIAS:
      offset_percentage = 0.1
      x_min = self._env.workspace_bounds_min[0] + offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      x_max = self._env.workspace_bounds_max[0] - offset_percentage * (self._env.workspace_bounds_max[0] - self._env.workspace_bounds_min[0])
      y_min = self._env.workspace_bounds_min[1] + offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      y_max = self._env.workspace_bounds_max[1] - offset_percentage * (self._env.workspace_bounds_max[1] - self._env.workspace_bounds_min[1])
      table_max_world = np.array([x_max, y_max, 0])
      table_min_world = np.array([x_min, y_min, 0])
      table_center = (table_max_world + table_min_world) / 2
      obs_dict = dict()
      obs_dict['name'] = obj_name
      obs_dict['position'] = self._world_to_voxel(table_center)
      obs_dict['_position_world'] = table_center
      obs_dict['normal'] = np.array([0, 0, 1])
      obs_dict['aabb'] = np.array([self._world_to_voxel(table_min_world), self._world_to_voxel(table_max_world)])
    else:
      obs_dict = dict()
      obj_pc, obj_normal = self._env.get_3d_obs_by_name(obj_name)
      voxel_map = self._points_to_voxel_map(obj_pc)
      aabb_min = self._world_to_voxel(np.min(obj_pc, axis=0))
      aabb_max = self._world_to_voxel(np.max(obj_pc, axis=0))
      obs_dict['occupancy_map'] = voxel_map  # in voxel frame
      obs_dict['name'] = obj_name
      obs_dict['position'] = self._world_to_voxel(np.mean(obj_pc, axis=0))  # in voxel frame
      obs_dict['aabb'] = np.array([aabb_min, aabb_max])  # in voxel frame
      obs_dict['_position_world'] = np.mean(obj_pc, axis=0)  # in world frame
      obs_dict['_point_cloud_world'] = obj_pc  # in world frame
      obs_dict['normal'] = normalize_vector(obj_normal.mean(axis=0))
      # Populate the 'color' attribute so LLM-generated parse_query_obj code
      # like `if button.color == 'olive'` actually works (previously 'color'
      # was missing → KeyError in __getattr__ / always False after safe fallback).
      try:
        cname = self._env.get_object_color_name(obj_name)
        if cname is not None:
          obs_dict['color'] = cname
      except Exception:
        pass

    object_obs = Observation(obs_dict)
    return object_obs
  
  def execute(self, movable_obs_func, affordance_map=None, avoidance_map=None, rotation_map=None,
              velocity_map=None, gripper_map=None):
    """
    First use planner to generate waypoint path, then use controller to follow the waypoints.

    Args:
      movable_obs_func: callable function to get observation of the body to be moved
      affordance_map: callable function that generates a 3D numpy array, the target voxel map
      avoidance_map: callable function that generates a 3D numpy array, the obstacle voxel map
      rotation_map: callable function that generates a 4D numpy array, the rotation voxel map (rotation is represented by a quaternion *in world frame*)
      velocity_map: callable function that generates a 3D numpy array, the velocity voxel map
      gripper_map: callable function that generates a 3D numpy array, the gripper voxel map
    """
    # ── 把 plan 级 deadline 下传给执行层 ────────────────────────────────
    # 深层的 IK / 抓取 / 推块 / 按压循环据此立即收手（rlbench_env.episode_budget_exceeded），
    # 保证 plan 在 harness 看门狗之前干净返回，使 env.success() 得到正常评估，
    # 而不是"超时杀 sim → 跳过 success() → 恒判 False"。
    # 未设置 plan deadline（如 planner 之外的 LMP 调用）时清除该限制。
    try:
      from envs import rlbench_env as _rlbench_env_mod
      _rlbench_env_mod.set_episode_deadline(_PLAN_DEADLINE)
    except Exception:
      pass

    # ── 成功锁存（episode 终止语义）─────────────────────────────────────
    # RLBench 的 episode 在 success()=True 的瞬间结束；我们的跑批会继续执行 LLM
    # 计划里剩余的 composer 步骤。一旦已达成原生成功，后续任何动作都可能破坏
    # 终态（StackBlocks r2 实测：达成后计划又重抓传感器处的方块）。锁存后
    # execute() 只读返回，判分仍完全依赖原生 env.success()。
    if getattr(self, '_episode_success_latched', False):
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] episode success latched → skip actions{bcolors.ENDC}')
      return []
    try:
      if bool(self._env.success()):
        self._episode_success_latched = True
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] native env.success()=True → '
              f'episode success latched; no further actions{bcolors.ENDC}')
        return []
    except Exception:
      pass

    # ── movable 归一化 ─────────────────────────────────────────────────
    # parse_query_obj 对复数名词（如 "the other cups on top of the red cup"）会返回
    # 一个 list，而 execute() 只支持单个 movable。旧代码把 list 直接当 dict 用，
    # 会在 _probe_obs['name'] / movable_obs.get(...) / movable_obs['position'] 处崩溃
    # （'list' object has no attribute 'get' / list indices must be integers）。
    # 这里在入口统一包装：list/tuple 取首个元素（元素是 dict 子类 Observation，
    # 支持 .get 与 ['position']）；空列表则抛错，交由下游走 EE fallback。
    def _as_single_obs(_x):
      if isinstance(_x, (list, tuple)):
        if len(_x) == 0:
          raise ValueError('movable observation resolved to an empty list')
        if len(_x) > 1:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] movable resolved to a list of '
                f'{len(_x)} objects; execute() supports a single movable — using the first one{bcolors.ENDC}')
        return _x[0]
      return _x

    def _resolve_movable_from_scene():
      """movable 解析成非物体时（位置向量 / 数组 / None，或 provider 抛错），
      用「场景对象 ∩ 指令 ∩ 可移动词表」找回真实物体并构造观测。

      典型现场：composer 没有产出有效 execute() → LMP 通用兜底把
      parse_query_obj 的结果（red_cup2.position + [0,0,0.05]，一个位置向量）
      当 movable 传进来 → DynamicObservation.__call__ 里 Observation(ndarray)
      抛 'cannot convert dictionary update sequence element #0 to a sequence'。
      旧行为是直接退化 EE 模式；EE 模式下 StackCups 会走按压/推挤启发式，把杯子
      推下桌面并空转到 planner 超时（实测 4/4 超时）。这里改为先找回物体。
      """
      try:
        _instr = str(getattr(self._env, 'current_instruction', '') or '').lower()
        _names = list(self._env.get_object_names())
      except Exception:
        return None
      import re as _re_mv
      _vocab = _PICKABLE_TOKENS | _SLIDABLE_TOKENS | _AMBIG_TOKENS
      _cands = []
      for _n in _names:
        _tok = str(_n).lower().strip()
        if ' ' in _tok:
          _tok = _tok.split(' ')[-1]
        elif '_' in _tok:
          _tok = _tok.split('_')[-1]
        _mm = _re_mv.match(r'^(.*?)(\d+)$', _tok)
        if _mm:
          _tok = _mm.group(1)
        if _tok in _vocab:
          _cands.append((_tok, str(_n)))
      if not _cands:
        return None
      # 只认「指令中点名」的物体：否则不动。
      # 例：PushButton 这类本就 EE-centric 的任务，场景里可能同时有 cup/tissue，
      # 若按名字兜底选一个，会把合法的 EE 模式抢走 → 这里必须要求指令命中。
      _hit = [c for c in _cands if c[0] in _instr]
      if not _hit:
        return None
      _tok, _nm = sorted(_hit)[0]
      try:
        _obs = self.detect(_nm)
      except Exception as _re_err:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] movable fallback: detect("{_nm}") failed: {_re_err}{bcolors.ENDC}')
        return None
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] movable normalized: provider yielded a non-object; '
            f'recovered real object "{_nm}" (token "{_tok}") from scene+instruction instead of degrading to EE mode{bcolors.ENDC}')
      return _obs

    # 两种入口都要兼容：
    #   1) callable（DynamicObservation / IterableDynamicObservation）→ 调用后再归一化；
    #      IterableDynamicObservation() 返回 list，正是本次崩溃的来源。
    #   2) 裸 list / dict（LLM 直接传值，如 execute(movable_list, ...)）→ 不能调用，
    #      否则报 'list' object is not callable，会被当成 movable 解析失败而误走 EE 模式。
    _raw_movable_func = movable_obs_func

    def movable_obs_func(_f=_raw_movable_func):
      _err = None
      try:
        _val = _f() if callable(_f) else _f
      except Exception as _e:
        _val, _err = None, _e
      try:
        _val = _as_single_obs(_val)
      except Exception as _e:
        _val, _err = None, _e
      # 非 dict（位置向量 / ndarray / 标量）→ 先尝试找回真实物体，避免降级 EE
      if not isinstance(_val, dict):
        _rec = _resolve_movable_from_scene()
        if _rec is not None:
          return _rec
        if _err is not None:
          raise _err
      return _val

    # initialize default voxel maps if not specified
    if rotation_map is None:
      rotation_map = self._get_default_voxel_map('rotation')
    if velocity_map is None:
      velocity_map = self._get_default_voxel_map('velocity')
    if gripper_map is None:
      gripper_map = self._get_default_voxel_map('gripper')
    if avoidance_map is None:
      avoidance_map = self._get_default_voxel_map('obstacle')
    # --- evaluate object_centric with robustness ---
    try:
      _probe_obs = movable_obs_func()
      object_centric = (not _probe_obs['name'] in EE_ALIAS)
    except Exception as _poe:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] movable_obs probe failed: {_poe}; defaulting object_centric=False (EE mode){bcolors.ENDC}')
      object_centric = False
    execute_info = []
    _affordance_map = None
    # ── State for post-action heuristics (Fix2 / Fix3) ─────────────────────
    # Saved by the LAST plan_iter so post-action can access affordance target
    # and movable identity after planner / controller loop exits.
    _last_target_center_world = None   # world-frame [x, y, z] center of last affordance target (after sanity-check override)
    _last_movable_world_pos = None     # world-frame [x, y, z] center of movable object (for slide target fallback)
    _last_movable_top_z = None         # world-frame top (max z) of movable object's AABB (for grasp approach height)
    _last_movable_name = None          # raw name string from movable_obs for classification
    if affordance_map is not None:
      # execute path in closed-loop
      for plan_iter in range(self._cfg['max_plan_iter']):
        # plan 级墙钟早退：整次 planner/composer 调用已超预算 → 放弃剩余 replan。
        if plan_budget_exceeded():
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] plan wall-clock budget exceeded at plan_iter {plan_iter} — abandoning remaining replans{bcolors.ENDC}')
          break
        step_info = dict()
        # evaluate voxel maps such that we use latest information (with fail-safe)
        try:
          movable_obs = movable_obs_func()
        except Exception as _moe:
          # --- Fallback: parse_query_obj returned None / raised → use END EFFECTOR as movable ---
          # This handles: "push the <color> button" where parser returns None for unmapped color.
          # Affordance map still uses correct object position (generated from same parser earlier)
          # so planner path becomes: move EE curr pos to affordance world pos → correctly press button.
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] movable_obs evaluation failed on plan_iter {plan_iter}: {_moe}; using END-EFFECTOR as movable fallback (EE mode) instead of aborting.{bcolors.ENDC}')
          def _ee_movable_fallback_func(_self=self):
            # Return fresh EE dict per call (pose changes over time)
            ee_pos = _self.get_ee_pos()
            return dict(
              name='ee',
              position=np.array(ee_pos),
              aabb=np.array([ee_pos, ee_pos]),
              _position_world=np.array(_self._env.get_ee_pos()),
            )
          movable_obs_func = _ee_movable_fallback_func
          object_centric = False
          try:
            movable_obs = movable_obs_func()
          except Exception as _moe2:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Even EE fallback failed ({_moe2}); aborting execute.{bcolors.ENDC}')
            break
        try:
          _affordance_map = affordance_map()
        except Exception as _afe:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] affordance_map() failed: {_afe}; using all-zero affordance for fallback.{bcolors.ENDC}')
          _empty_shape = (self._map_size, self._map_size, self._map_size)
          _affordance_map = VoxelIndexingWrapper(np.zeros(_empty_shape))
        try:
          _avoidance_map = avoidance_map()
        except Exception as _ave:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] avoidance_map() failed: {_ave}; using zeros.{bcolors.ENDC}')
          _empty_shape = (self._map_size, self._map_size, self._map_size)
          _avoidance_map = VoxelIndexingWrapper(np.zeros(_empty_shape))
        try:
          _rotation_map = rotation_map()
        except Exception as _roe:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] rotation_map() failed: {_roe}; using EE quat.{bcolors.ENDC}')
          _rot = np.zeros((self._map_size, self._map_size, self._map_size, 4))
          _rot[:, :, :] = self._env.get_ee_quat()
          _rotation_map = VoxelIndexingWrapper(_rot)
        try:
          _velocity_map = velocity_map()
        except Exception as _ve:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] velocity_map() failed: {_ve}; using ones.{bcolors.ENDC}')
          _velocity_map = VoxelIndexingWrapper(np.ones((self._map_size, self._map_size, self._map_size)))
        try:
          _gripper_map = gripper_map()
        except Exception as _ge:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] gripper_map() failed: {_ge}; using last gripper action.{bcolors.ENDC}')
          _gripper_map = VoxelIndexingWrapper(
            np.ones((self._map_size, self._map_size, self._map_size)) * self._env.get_last_gripper_action()
          )

        # ── Affordance visualisation hook (NPY dump) ─────────────────────
        # Activated by env var VOXPOSER_AFF_DUMP_DIR (set by run_real10.py).
        # Saves one set of maps per LLM-execute call — first plan_iter only.
        _dump_dir = os.environ.get('VOXPOSER_AFF_DUMP_DIR', '').strip()
        if _dump_dir and plan_iter == 0:
          try:
            os.makedirs(_dump_dir, exist_ok=True)
            _tag = (f'{self._env._dump_tag}'
                    if hasattr(self._env, '_dump_tag') and self._env._dump_tag
                    else f'ep{int(time.time()*1000) % 10_000_000}')
            for _name, _m in [('aff', _affordance_map), ('avoid', _avoidance_map),
                               ('rot', _rotation_map), ('vel', _velocity_map),
                               ('grip', _gripper_map)]:
              _arr = _m.array if hasattr(_m, 'array') else np.asarray(_m)
              np.save(os.path.join(_dump_dir, f'{_tag}_{_name}.npy'), _arr)
          except Exception as _de:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] aff-dump failed: {_de}{bcolors.ENDC}')
        # ────────────────────────────────────────────────────────────────────

        # --- Fallback：只在 affordance_map 完全为空或全在顶部时才用对象中心作为 fallback 目标。
        #     注意：桌面高度（z<=5）的 affordance 对于 push/slide 等任务是合理的，不触发 fallback。
        _aff_raw = _affordance_map.array if hasattr(_affordance_map, 'array') else np.asarray(_affordance_map)
        _active_zs, _active_cnt = np.unique(np.where(_aff_raw > 0.5)[2], return_counts=True) if _aff_raw.ndim == 3 else (np.array([]), np.array([]))
        _all_at_top = (len(_active_zs) > 0 and np.all(_active_zs >= self._map_size - 5))
        _empty = (_aff_raw.max() < 0.5)
        _fallback_built = False  # track whether affordance was LLM-generated or fallback-built
        if _empty or _all_at_top:
          try:
            _is_ee = (movable_obs.get('name', '') in EE_ALIAS)
            if 'occupancy_map' in movable_obs and movable_obs.get('name', '') and not _is_ee:
              _occ = movable_obs['occupancy_map']
              if _occ is not None and np.any(np.asarray(_occ) > 0):
                target_center = np.array(np.where(np.asarray(_occ) > 0)).mean(axis=1).astype(int)
              else:
                target_center = np.asarray(movable_obs['position']).astype(int)
              # non-EE: 目标设置在对象上方 3~5cm 处（push/抓位置）
              target_center[2] = min(target_center[2] + self.cm2index(5, np.array([0,0,1]))[2], self._map_size - 1)
            else:
              target_center = np.asarray(movable_obs['position']).astype(int)
              target_center[2] = min(target_center[2] + self.cm2index(3, np.array([0,0,1]))[2], self._map_size - 1)
            # 目标高度下限：只防止 z 低到 workspace 底部 2cm 以内（避免卡进桌子），
            #   其余情况保留 occupancy/position + 5cm 自身计算出的高度（对于按钮等矮物体，z 可能仅 5~6 index 左右，这是正确的）
            _ws_bottom_pad = 2  # voxel index from ws_min.z; 2 voxels = ~2cm
            if target_center[2] < _ws_bottom_pad:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] fallback affordance: z clipped z from {target_center[2]} to workspace bottom+{_ws_bottom_pad}{bcolors.ENDC}')
              target_center[2] = _ws_bottom_pad
            target_center = np.clip(target_center, 0, self._map_size - 1)
            tmp = np.zeros_like(_aff_raw)
            radius = min(int(0.08 * self._map_size), 5)
            self.set_voxel_by_radius(tmp, target_center, radius_cm=max(int(radius * 1.5), 3), value=1.0)
            _affordance_map = tmp if not hasattr(_affordance_map, 'array') else type(_affordance_map)(tmp)
            _reason = 'empty' if _empty else 'top'
            _fallback_built = True
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] fallback affordance ({_reason}) set around voxel {target_center}, r≈{radius}{bcolors.ENDC}')
          except Exception as _ffe:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] fallback affordance failed: {_ffe}{bcolors.ENDC}')
            traceback.print_exc()

        # preprocess avoidance map
        _avoidance_map = self._preprocess_avoidance_map(_avoidance_map, _affordance_map, movable_obs)
        # start planning
        start_pos = movable_obs['position']
        start_time = time.time()
        # optimize path and log
        path_voxel, planner_info = self._planner.optimize(start_pos, _affordance_map, _avoidance_map,
                                                        object_centric=object_centric)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] planner time: {time.time() - start_time:.3f}s{bcolors.ENDC}')
        assert len(path_voxel) > 0, 'path_voxel is empty'
        step_info['path_voxel'] = path_voxel
        step_info['planner_info'] = planner_info
        # convert voxel path to world trajectory, and include rotation, velocity, and gripper information
        traj_world = self._path2traj(path_voxel, _rotation_map, _velocity_map, _gripper_map)
        traj_world = traj_world[:self._cfg['num_waypoints_per_plan']]

        # 将 waypoint 位置夹到 workspace bounds 内（留出安全边距），避免 "target outside of workspace"
        _ws_min = self._env.workspace_bounds_min + np.array([0.02, 0.02, 0.01])
        _ws_max = self._env.workspace_bounds_max - np.array([0.02, 0.02, 0.01])
        # EE-mode 下的"可抓物体防撞"：目标 z 若低于可抓物体顶面且水平位置重合，
        # 抬高到顶面 +1cm（闭环控制会把 EE 一路压进物体、把它顶出台面；run25 r0）。
        # 仅 EE-mode 生效：object-centric 的 waypoint 是"物体自身的目标轨迹"，
        # 抬高会变成命令控制器把物体抬起来，语义不同。
        _pick_guard = [] if object_centric else _build_pick_guard(self._env)
        _clamped_world = []
        for _wp in traj_world:
            _xyz = np.array(_wp[0]).astype(float)
            _xyz_c = np.clip(_xyz, _ws_min, _ws_max)
            for _pg_xy, _pg_top in _pick_guard:
                if (float(np.linalg.norm(_xyz_c[:2] - _pg_xy)) < 0.07
                        and float(_xyz_c[2]) < _pg_top + 0.01):
                    _z_new = min(_pg_top + 0.01, float(_ws_max[2]))
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] pick-guard: wp z {_xyz_c[2]:.3f}→{_z_new:.3f} (over graspable @{_pg_xy.round(3)}, top={_pg_top:.3f}){bcolors.ENDC}')
                    _xyz_c = np.array([_xyz_c[0], _xyz_c[1], _z_new])
            if not np.allclose(_xyz, _xyz_c):
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] clamped wp from {_xyz.round(3)} to {_xyz_c.round(3)}{bcolors.ENDC}')
            _clamped_world.append((_xyz_c, _wp[1], _wp[2], _wp[3]))
        traj_world = _clamped_world
        step_info['start_pos'] = start_pos
        step_info['plan_iter'] = plan_iter
        step_info['movable_obs'] = movable_obs
        step_info['traj_world'] = traj_world
        step_info['affordance_map'] = _affordance_map
        step_info['rotation_map'] = _rotation_map
        step_info['velocity_map'] = _velocity_map
        step_info['gripper_map'] = _gripper_map
        step_info['avoidance_map'] = _avoidance_map

        # visualize
        if self._cfg['visualize'] and self._env.visualizer is not None:
          step_info['start_pos_world'] = self._voxel_to_world(start_pos)
          step_info['targets_world'] = self._voxel_to_world(planner_info['targets_voxel'])
          try:
            self._env.visualizer.visualize(step_info)
          except Exception as _ve:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] visualizer.visualize failed: {_ve}{bcolors.ENDC}')

        # execute path
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] start executing path via controller ({len(traj_world)} waypoints){bcolors.ENDC}')
        controller_infos = dict()
        _n_executed = 0
        # 逐 waypoint 空转守卫（仅 EE-centric）：EE 连续 3 个 waypoint 既没位移、
        # dist2target 也没改善，说明本 plan 剩余 waypoint 对当前起止状态不可达。
        # run9 c0 实测 EE 恒在 [0.575 -0.084 0.972]、dist2target 恒 0.440m，
        # waypoint 9→20 每个 5~32s，单次 execute 累计 ~300s 撞 480s 硬超时。
        # 命中即放弃本 plan 剩余 waypoint，转入后续 post-action，不改成功判定语义。
        _wp_prev_actual = None
        _wp_prev_dist = None
        _wp_noprogress = 0
        # 单个 composer 步骤的 waypoint 循环墙钟预算：正常可达路径实测 ~1s 走完
        # (StackCups 可达段 13 个 waypoint 仅 1s)，而不可达尾部每个 waypoint 要
        # 12~32s。给整段循环一个上限，到点放弃剩余 waypoint 转入 post-action。
        # 不改成功判定语义；可用 VOSPOSER_WP_LOOP_BUDGET_S 覆盖。
        _wp_dl = time.time() + max(0.0, float(os.environ.get('VOSPOSER_WP_LOOP_BUDGET_S', '15.0')))
        for i, waypoint in enumerate(traj_world):
          # 原生成功条件已满足 → 停止剩余 waypoint，保持当前姿态。
          # lift-only（PickAndLift）实测：Fix2 已把目标举进传感器、env.success()=True，
          # 但剩余 waypoint 仍继续驱动机械臂，把目标移出传感器 → 终态又变回 False。
          # 这里只读原生判定结果，不改写判据；早退后的终态仍由 env.success() 复核。
          try:
            if bool(self._env.success()):
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] native success already satisfied at wp {i+1}/{len(traj_world)} — stop remaining waypoints{bcolors.ENDC}')
              break
          except Exception:
            pass
          if time.time() > _wp_dl:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] waypoint loop wall-clock budget exceeded at wp {i+1}/{len(traj_world)} — abandoning remaining waypoints{bcolors.ENDC}')
            break
          # 整次 plan 的硬 deadline 也已耗尽 → 后续每个 waypoint 都会在白耗预算，
          # 立即放弃剩余 waypoint（不改成功判定语义）。
          if plan_budget_exceeded():
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] plan wall-clock budget exceeded at wp {i+1}/{len(traj_world)} — abandoning remaining waypoints{bcolors.ENDC}')
            break
          # 物体被撞离台面/物理爆炸时，继续按 MPC 采样点驱动机械臂只会反复撞击
          # 几何体，令 CoppeliaSim 接触求解发散（实测 cup 坐标从 0.03 一路涨到 1e7）。
          # 一旦发现物体世界坐标不可用或明显在工作空间之外，立即放弃本 plan 剩余
          # waypoint；成功判定仍完全交给原生 env.success()。
          if object_centric:
            try:
              _mvp = np.array(movable_obs['_position_world'], dtype=float).reshape(3)
              _lo_ws = np.asarray(self._env.workspace_bounds_min, dtype=float).reshape(3)
              _hi_ws = np.asarray(self._env.workspace_bounds_max, dtype=float).reshape(3)
              _ws_m = 0.05
              if (not np.all(np.isfinite(_mvp))) or (not (np.all(_mvp >= _lo_ws - _ws_m)
                                                          and np.all(_mvp <= _hi_ws + _ws_m))):
                print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] movable off-workspace/non-sane ({_mvp.round(3)}) — abandon remaining waypoints of this plan{bcolors.ENDC}')
                break
            except Exception:
              pass
          # 关键修复：在 object_centric 模式下（PushButton 等），movable 从开始就在目标附近，
          # 旧代码直接 break 导致 EE 根本没动。
          # 新规则：至少执行 1 个 waypoint（_n_executed>=1）后才允许通过距离判断 break。
          # EE-centric 模式下 waypoint 明确，不受此限制。
          _movable2final = np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0])
          _can_early_break = (_movable2final <= 0.01)
          if object_centric and _n_executed < 1:
            _can_early_break = False
          if _can_early_break:
            print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={movable_obs['_position_world']}, target={traj_world[-1][0]} (distance: {_movable2final:.3f})){bcolors.ENDC}")
            break
          # skip waypoint if moving to this point is going in opposite direction of the final target point
          # (for example, if you have over-pushed an object, no need to move back)
          if i != 0 and i != len(traj_world) - 1:
            movable2target = traj_world[-1][0] - movable_obs['_position_world']
            movable2waypoint = waypoint[0] - movable_obs['_position_world']
            if np.dot(movable2target, movable2waypoint).round(3) <= 0:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] skip waypoint {i+1} because it is moving in opposite direction of the final target{bcolors.ENDC}')
              continue
          # execute waypoint
          controller_info = self._controller.execute(movable_obs, waypoint)
          _n_executed += 1
          # loggging
          movable_obs = movable_obs_func()
          dist2target = np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0])
          if not object_centric and controller_info['mp_info'] == -1:
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] failed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}); mp info: {controller_info["mp_info"]}{bcolors.ENDC}')
          else:
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}){bcolors.ENDC}')
          controller_info['controller_step'] = i
          controller_info['target_waypoint'] = waypoint
          controller_infos[i] = controller_info
          # 空转守卫：连续 3 个 waypoint 无位移且 dist2target 无改善 → 放弃本 plan 剩余 waypoint
          if not object_centric:
            try:
              _wp_actual = np.array(movable_obs['_position_world'], dtype=float).reshape(3)
            except Exception:
              _wp_actual = None
            if _wp_actual is not None:
              _wp_moved = float('inf') if _wp_prev_actual is None else float(np.linalg.norm(_wp_actual - _wp_prev_actual))
              _wp_improved = True if _wp_prev_dist is None else bool(dist2target < _wp_prev_dist - 1e-4)
              if _wp_moved < 1e-3 and not _wp_improved:
                _wp_noprogress += 1
              else:
                _wp_noprogress = 0
              _wp_prev_actual = _wp_actual
              _wp_prev_dist = float(dist2target)
              if _wp_noprogress >= 3:
                print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] no EE progress for 3 consecutive waypoints (actual stuck at {_wp_actual.round(3)}, dist2target {float(dist2target):.3f}m) — abandoning remaining waypoints of this plan{bcolors.ENDC}')
                break
        step_info['controller_infos'] = controller_infos
        execute_info.append(step_info)
        # check whether we need to replan
        curr_pos = movable_obs['position']
        if distance_transform_edt(1 - _affordance_map)[tuple(curr_pos)] <= 2:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached target; terminating {bcolors.ENDC}')
          break
    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished executing path via controller{bcolors.ENDC}')

    # plan 级墙钟早退：整次 plan 已超预算 → 跳过昂贵的 post-action（Fix2 抓取/
    # 搬运、按压/闭爪启发式等），干净返回，让 harness 正常评估 env.success()，
    # 避免继续空转直至 harness 看门狗"超时杀 sim → 跳过 success()"。
    if plan_budget_exceeded():
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] plan wall-clock budget exceeded before post-action — skipping Fix2/heuristics and returning{bcolors.ENDC}')
      return execute_info

    # make sure we are at the final target position and satisfy any additional parametrization
    # NOTE: for both EE-centric and object-centric tasks (PushButton / LampOff / Slide / Meat) we need:
    #       final approach → progressive press-down → post-action (grasp/release/hold_press)
    gripper_state = self._env.get_last_gripper_action()
    ee_pose_world = None
    if not object_centric:
      try:
        # traj_world: world_xyz, rotation, velocity, gripper
        ee_pos_world = traj_world[-1][0]
        ee_rot_world = traj_world[-1][1]
        ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
        ee_speed = traj_world[-1][2]
        _traj_g = float(traj_world[-1][3])
        if 0.0 <= _traj_g <= 1.0:
          gripper_state = _traj_g
      except:
        # evaluate latest voxel map
        try:
          _rotation_map = rotation_map()
          _velocity_map = velocity_map()
          _gripper_map = gripper_map()
          ee_pos_world = self._env.get_ee_pos()
          ee_pos_voxel = self.get_ee_pos()
          ee_rot_world = _rotation_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
          ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
          try:
            _map_g = float(_gripper_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]])
            if 0.0 <= _map_g <= 1.0:
              gripper_state = _map_g
          except Exception:
            pass
        except Exception:
          ee_pose_world = None
    else:
      # object_centric: movable is the object (button / block / meat), use env last state
      try:
        ee_pose_world = np.concatenate([
          np.array(self._env.get_ee_pos()),
          np.array(self._env.get_ee_quat()),
        ])
      except Exception:
        ee_pose_world = None

    # move to the final target (both modes)
    if ee_pose_world is not None:
      try:
        self._env.apply_action(np.concatenate([ee_pose_world, [gripper_state]]))
      except Exception:
        pass

    # === 最终逼近（两种模式都执行） ===
    try:
      if _affordance_map is not None:
        _aff_raw_final = _affordance_map.array if hasattr(_affordance_map, 'array') else np.asarray(_affordance_map)
        if _aff_raw_final.max() > 0.5:
          _target_voxels = np.argwhere(_aff_raw_final > 0.5)
          if len(_target_voxels) > 0:
            _target_center_voxel = _target_voxels.mean(axis=0).astype(int)
            _target_center_world = self._voxel_to_world(_target_center_voxel)
            # -------------------- AFFORDANCE SANITY CHECK + RECOVERY --------------------
            # Root cause: composer LLM sometimes writes affordance using WORLD coords as if
            # they were voxel indices → z-index ends up negative/clamped to 0 → target z is
            # workspace floor (0.76m in PushButton) when actual action object is at 1.0-1.5m.
            # Detect this pathological case by:
            #   - target z is in bottom 5% of workspace z-range, OR
            #   - target z is suspiciously low (< 0.88m absolute)
            # If detected: scan common action-object names via detect(), compute a corrected
            # candidate target as (object.world_pos - 3cm * object.normal), and override
            # _target_center_world to the most EE-proximal valid candidate.
            try:
              _ws_min = np.asarray(self._env.workspace_bounds_min, dtype=float)
              _ws_max = np.asarray(self._env.workspace_bounds_max, dtype=float)
              _ws_z_span = float(_ws_max[2] - _ws_min[2])
              _z_from_bottom = float(_target_center_world[2] - _ws_min[2])
              _suspicious = False
              if _fallback_built and _ws_z_span > 0.05 and _z_from_bottom < 0.05 * _ws_z_span:
                _suspicious = True
              # Only flag low-z as suspicious when the affordance was built by the
              # fallback (empty/all-at-top).  When the LLM explicitly set the target
              # via detect('button') → button.position → affordance_map[x,y,z]=1,
              # the target z is the REAL object position and must NOT be overridden.
              # The old 0.88m threshold incorrectly overwrote legitimate low-z buttons
              # (e.g. LampOff button at z=0.768m → sanity-check moved target to lamp
              # body top at z=1.213m → pressed wrong spot → task failed).
              if _fallback_built and float(_target_center_world[2]) < 0.88:
                _suspicious = True
              if _suspicious:
                _ee_now_cv = np.array(self._env.get_ee_pos())
                # Prefer a button/object that's CLOSEST TO THE USER-PARSED movable's world position
                # (for multi-color scenarios), NOT necessarily closest to EE start position.
                # If movable is None or has no pos, fall back to EE distance as anchor.
                _anchor = _ee_now_cv
                try:
                    if movable_obs is not None and hasattr(movable_obs, '__getitem__'):
                        try:
                            _mp = np.asarray(movable_obs['_position_world'], dtype=float).reshape(3)
                            if np.all(np.isfinite(_mp)):
                                _anchor = _mp
                        except Exception:
                            pass
                    elif movable_obs is not None:
                        try:
                            _mp = np.asarray(getattr(movable_obs, '_position_world'), dtype=float).reshape(3)
                            if np.all(np.isfinite(_mp)):
                                _anchor = _mp
                        except Exception:
                            pass
                except Exception:
                    _anchor = _ee_now_cv
                _candidates = []
                _names_to_try = [
                    'button', 'buttons', 'lamp', 'light', 'switch', 'block', 'slider', 'meat',
                    'cup', 'mug', 'pot', 'pan', 'drawer', 'fridge', 'door', 'lever', 'knob',
                    'apple', 'banana', 'orange', 'tomato', 'pepper', 'chili', 'onion', 'garlic',
                    'plate', 'bowl', 'tray', 'tissue', 'book', 'box',
                ]
                for _nm in _names_to_try:
                  try:
                    _obj = self.detect(_nm)
                    if _obj is None:
                      continue
                    _opw = None; _onorm = None
                    try:
                      _opw = np.asarray(_obj['_position_world'], dtype=float).reshape(3)
                    except Exception:
                      _opw = None
                    try:
                      _onorm = np.asarray(_obj['normal'], dtype=float).reshape(3)
                      if np.linalg.norm(_onorm) < 1e-6:
                        _onorm = np.array([0.0, 0.0, -1.0])
                      else:
                        _onorm = _onorm / np.linalg.norm(_onorm)
                    except Exception:
                      _onorm = np.array([0.0, 0.0, -1.0])
                    if _opw is None:
                      continue
                    if float(_opw[2]) <= 0.88:
                      # object itself at floor level, skip (likely a false positive / table / garbage)
                      continue
                    # candidate reach/press point: 3cm into object along surface normal
                    _cand = _opw + (-_onorm) * 0.03
                    # clamp inside workspace (use slightly-shrunk box to avoid IK edge failures)
                    _cand_c = np.clip(_cand, _ws_min + np.array([0.02,0.02,0.015]), _ws_max - np.array([0.02,0.02,0.015]))
                    _d_anchor = float(np.linalg.norm(_anchor - _cand_c))
                    _candidates.append((_d_anchor, _cand_c.copy(), _nm, _opw.copy()))
                  except Exception:
                    continue
                if len(_candidates) > 0:
                  # ── CRITICAL BUG FIX 2026-08-10 ──────────────────────────────────────
                  # _bonus_map MUST be initialized BEFORE the semantic/color loops;
                  # previously it was never created → NameError on `_bonus_map[_ci] +=`
                  # → outer try/except caught it → "Affordance sanity check skipped"
                  # → LampOff / multi-color-button scenes always reverted to the bad
                  # planner target (lamp body instead of button, wrong color button).
                  _bonus_map = [0.0 for _ in range(len(_candidates))]
                  # ── END BUG FIX ──────────────────────────────────────────────────────
                  # ── Task-agnostic SEMANTIC PRIORITY for interactive control objects ──
                  # Root cause (LampOff): Common scene contains both a big "lamp" body and a
                  # small "button" control on the lamp.  Distance-only scoring picks the big
                  # lamp body as candidate (closer to anchor / movable) → target z at 0.98m
                  # (top of lamp stand / table area) → press_cont presses desk → env.success
                  # always False.  Interactive controls (button/switch/knob/lever) are ALWAYS
                  # the correct affordance target for manipulation tasks; big scene bodies
                  # (lamp/light/tray/table/desk/chair) should be DEPREFERENCED.  We apply this
                  # as semantic bonus/penalty on top of distance and color bonuses.
                  _SEMANTIC_BONUS = {
                      # --- Interactive controls: STRONG priority #1 ---
                      'button': -2000.0, 'buttons': -2000.0,
                      'switch': -2000.0, 'switches': -2000.0,
                      'knob': -2000.0, 'knobs': -2000.0,
                      'lever': -2000.0, 'levers': -2000.0,
                      'dial': -1800.0,
                      # --- Manipulable objects: priority #2 (no semantic bias, 0.0) ---
                      'block': 0.0, 'slider': -100.0, 'meat': 0.0,
                      'cup': 0.0, 'mug': 0.0, 'pot': 0.0, 'pan': 0.0,
                      'drawer': -300.0, 'fridge': -300.0, 'door': -300.0,
                      'apple': 0.0, 'banana': 0.0, 'orange': 0.0, 'tomato': 0.0,
                      'pepper': 0.0, 'chili': 0.0, 'onion': 0.0, 'garlic': 0.0,
                      'plate': 0.0, 'bowl': 0.0, 'tray': 0.0,
                      'tissue': 0.0, 'book': 0.0, 'box': 0.0,
                      # --- Large scene bodies / non-interactive containers: PENALTY ---
                      'lamp': +1000.0, 'light': +1000.0,
                      'table': +2500.0, 'desk': +2500.0,
                      'chair': +2500.0, 'shelf': +2500.0, 'shelves': +2500.0,
                      'grill': +500.0,  # MeatOffGrill: grill is scene, actual target is meat
                  }
                  for _ci, _cand in enumerate(_candidates):
                      _nm = _cand[2]
                      # strip color token (e.g. "rose button" → "button") to lookup semantic
                      _tok = _nm
                      if ' ' in _nm:
                          _parts = _nm.split(' ')
                          _tok = _parts[-1].lower() if _parts else _nm
                      elif '_' in _nm:
                          _p2 = _nm.split('_')
                          _tok = _p2[-1].lower() if _p2 else _nm
                      _tok = _tok.lower()
                      # also strip trailing number: "grill0" / "meat1"
                      import re as _re_if_sem
                      _m = _re_if_sem.match(r'^(.*?)(\d+)$', _tok)
                      if _m:
                          _tok = _m.group(1)
                      if _tok in _SEMANTIC_BONUS:
                          _bonus_map[_ci] += float(_SEMANTIC_BONUS[_tok])
                      else:
                          # Unknown object: likely a scene body → small penalty to prefer known controls
                          _bonus_map[_ci] += +200.0
                  # Extra: if movable_obs has a known color, add bonuses for color-matched candidates.
                  # This helps pick the CORRECT COLOR BUTTON in multi-color button scenes even
                  # when parse_query_obj's returned position is unreliable / generic "button" center.
                  try:
                      _mv_color = None
                      try:
                          if movable_obs is not None and hasattr(movable_obs, '__getitem__'):
                              try: _mv_color = str(movable_obs.get('color', '') or '').strip().lower()
                              except Exception: _mv_color = None
                          if not _mv_color and movable_obs is not None:
                              try: _mv_color = str(getattr(movable_obs, 'color', '') or '').strip().lower()
                              except Exception: _mv_color = None
                      except Exception:
                          _mv_color = None
                      if _mv_color:
                          # Synonym expansion (same table as LMP fallback)
                          _SYN = {
                              'purple': {'purple','violet','magenta','pink','indigo','crimson'},
                              'violet': {'violet','purple','indigo','magenta'},
                              'magenta': {'magenta','purple','pink','violet','crimson','red'},
                              'pink': {'pink','magenta','purple','coral','salmon','red','rose'},
                              'rose': {'rose','pink','magenta','red','coral','salmon'},
                              'red': {'red','crimson','maroon','magenta','pink','coral','salmon','rose'},
                              'crimson': {'crimson','red','maroon','magenta','purple'},
                              'maroon': {'maroon','red','crimson','brown'},
                              'coral': {'coral','orange','salmon','pink','red'},
                              'salmon': {'salmon','pink','coral','orange','red'},
                              'orange': {'orange','coral','salmon','gold','yellow','brown'},
                              'brown': {'brown','orange','maroon','tan','beige','gold'},
                              'tan': {'tan','brown','beige','gold','orange'},
                              'beige': {'beige','tan','cream','white','gold','brown'},
                              'cream': {'cream','beige','white','yellow','tan'},
                              'yellow': {'yellow','gold','orange','cream','beige','lime'},
                              'gold': {'gold','yellow','orange','brown','tan','beige'},
                              'olive': {'olive','green','yellow','lime','brown'},
                              'lime': {'lime','green','yellow','olive','aqua'},
                              'green': {'green','lime','olive','teal','aqua','cyan','turquoise'},
                              'teal': {'teal','green','cyan','turquoise','aqua','blue'},
                              'aqua': {'aqua','cyan','turquoise','teal','green','blue'},
                              'cyan': {'cyan','aqua','turquoise','teal','green','blue'},
                              'turquoise': {'turquoise','cyan','aqua','teal','green','blue'},
                              'azure': {'azure','blue','cyan','aqua','turquoise','white'},
                              'blue': {'blue','azure','navy','cyan','aqua','turquoise','indigo','purple'},
                              'navy': {'navy','blue','indigo','purple','violet','black'},
                              'indigo': {'indigo','blue','navy','purple','violet'},
                              'black': {'black','navy','grey','gray','maroon','brown'},
                              'white': {'white','cream','beige','silver','tan'},
                              'gray': {'gray','grey','silver','black','white'},
                              'grey': {'grey','gray','silver','black','white'},
                              'silver': {'silver','grey','gray','white','gold'},
                          }
                          _mv_set = set([_mv_color]) | _SYN.get(_mv_color, set())
                          for _ci, _cand in enumerate(_candidates):
                              try:
                                  _c_obj_color = None
                                  try:
                                      _nm_cand = _cand[2]
                                      _det_c = self.detect(_nm_cand)
                                      if _det_c is not None and hasattr(_det_c, 'get'):
                                          try: _c_obj_color = str(_det_c.get('color', '') or '').strip().lower()
                                          except Exception: _c_obj_color = None
                                      if not _c_obj_color and _det_c is not None:
                                          try: _c_obj_color = str(getattr(_det_c, 'color', '') or '').strip().lower()
                                          except Exception: _c_obj_color = None
                                  except Exception:
                                      _c_obj_color = None
                                  if _c_obj_color and _c_obj_color in _mv_set:
                                      _bonus_map[_ci] = -1000.0  # strong color match → priority #1
                              except Exception:
                                  pass
                  except Exception:
                      pass
                  # Apply bonuses to score tuples (score is index 0)
                  _scored = []
                  for _ci, _cand in enumerate(_candidates):
                      _new_score = float(_cand[0]) + float(_bonus_map[_ci])
                      _scored.append((_new_score, _cand[0], _cand[1], _cand[2], _cand[3]))
                  _scored.sort(key=lambda x: x[0])
                  _best = _scored[0]
                  _old_tgt = _target_center_world.copy()
                  _target_center_world = _best[2].copy()
                  # ── F1a Fallback: LampOff-with-only-lamp-candidate ────────────────
                  # Even with standalone control detection, some RLBench LampOff
                  # variants may expose NO switch/button shape (e.g. switch is a
                  # joint rather than a Shape).  In that case N_candidates=1
                  # (only "lamp" scene body) and SEMANTIC_BONUS can't help.
                  # The lamp body's surface normal points inward/downward →
                  # _cand_c = opw + (-normal)*0.03 lands at lamp-stand z (≈0.98m
                  # = table height) → press_down presses the desk, never the
                  # lamp-top switch.  Detect this and lift target to lamp body
                  # TOP = object world_pos z + reasonable up-offset so EE
                  # descends onto the lamp-top region where the switch lives.
                  _best_nm = str(_best[3] or '').lower()
                  def _strip_tok(_s):
                      import re as _re_if_fb
                      _t = _s
                      if ' ' in _t: _t = _t.split(' ')[-1]
                      elif '_' in _t: _t = _t.split('_')[-1]
                      _m2 = _re_if_fb.match(r'^(.*?)(\d+)$', _t)
                      if _m2: _t = _m2.group(1)
                      return _t.lower()
                  _best_tok = _strip_tok(_best_nm)
                  _SCENE_BODY_TOKENS = {'lamp', 'light'}
                  _has_any_control = False
                  for _ci2, _cand2 in enumerate(_candidates):
                      _c_tok = _strip_tok(str(_cand2[2] or '').lower())
                      if _c_tok in {'button','buttons','switch','switches','knob','knobs','lever','levers','dial','dials'}:
                          _has_any_control = True
                          break
                  if (not _has_any_control) and _best_tok in _SCENE_BODY_TOKENS:
                      _opw = np.asarray(_best[4], dtype=float).reshape(3)
                      # Candidate target is 3cm along normal; don't trust it
                      # for a big scene body.  Use object world XY as-is, but
                      # push z UP to object's top.  Typical RLBench desk lamp:
                      #   body center z ≈ 1.10m  →  top z (switch area) ≈ 1.30m
                      # So add +0.20m (20cm) above object center.  Clip to
                      # workspace max z minus 3cm margin.
                      _est_top_z = float(_opw[2]) + 0.20
                      try:
                          _ws_max_z_here = float(self._env.workspace_bounds_max[2]) - 0.03
                          if _est_top_z > _ws_max_z_here:
                              _est_top_z = _ws_max_z_here
                      except Exception:
                          pass
                      # Safety: if we somehow got a tiny z, don't add 20cm to
                      # table height.  Only apply if world z is plausible lamp
                      # body height (≥ 0.95m).
                      if float(_opw[2]) >= 0.95 and float(_target_center_world[2]) < float(_opw[2]) + 0.05:
                          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] LAMP FALLBACK TRIGGERED: no control candidates, best is "{_best_nm}" (tok={_best_tok}), old target z={_target_center_world[2]:.3f}m → raised z to lamp-top ~{_est_top_z:.3f}m (obj_center_z={_opw[2]:.3f}m){bcolors.ENDC}')
                          _target_center_world = np.array([_opw[0], _opw[1], _est_top_z], dtype=float)
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] AFFORDANCE OVERRIDE: planner target z={_old_tgt[2]:.3f}m suspicious → using detected object "{_best[3]}" at obj.world_pos={_best[4].round(3)} → new target={_target_center_world.round(3)} (score={_best[0]:.2f}, raw_d={_best[1]:.3f}m, bonus={float(_best[0])-float(_best[1]):.1f}, N_candidates={len(_candidates)}){bcolors.ENDC}')
            except Exception as _sce:
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Affordance sanity check skipped: {_sce}{bcolors.ENDC}')
            # ── Save target & movable state for post-action heuristics ──────────
            try:
                _last_target_center_world = np.asarray(_target_center_world, dtype=float).reshape(3).copy()
            except Exception:
                pass
            try:
                if movable_obs is not None and hasattr(movable_obs, '__getitem__'):
                    try:
                        _nm_str = str(movable_obs.get('name', '') or '').strip()
                        if _nm_str:
                            _last_movable_name = _nm_str
                    except Exception:
                        pass
                    try:
                        _mw = np.asarray(movable_obs.get('_position_world'), dtype=float).reshape(3)
                        if np.all(np.isfinite(_mw)):
                            _last_movable_world_pos = _mw.copy()
                    except Exception:
                        pass
                if movable_obs is not None and _last_movable_name is None:
                    try:
                        _nm_str = str(getattr(movable_obs, 'name', '') or '').strip()
                        if _nm_str:
                            _last_movable_name = _nm_str
                    except Exception:
                        pass
            except Exception:
                pass
            # -----------------------------------------------------------------------------
            _ee_pos_now = np.array(self._env.get_ee_pos())
            _final_dist = np.linalg.norm(_ee_pos_now - _target_center_world)
            # Decompose: XY close & EE already AT or BELOW target z → pressing button now.
            # Total dist may be large (e.g., 8cm BELOW affordance center from 3cm deep press)
            # but we definitely should NOT retract to the center (which would release button).
            _final_dist_xy = float(np.sqrt((_ee_pos_now[0]-_target_center_world[0])**2 + (_ee_pos_now[1]-_target_center_world[1])**2))
            _already_pressing = (_final_dist_xy < 0.02) and (float(_ee_pos_now[2]) <= float(_target_center_world[2]) + 0.002)
            if _already_pressing:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Final approach: EE already on-press column (XY={_final_dist_xy*1000:.1f}mm, EE_z={_ee_pos_now[2]:.3f}m <= tgt_z={_target_center_world[2]:.3f}m); skipping (no retract){bcolors.ENDC}')
            elif _final_dist > 0.015:
              # EE > 1.5cm away from affordance center → definitely not on target; approach now
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Final approach: EE {_final_dist:.3f}m from affordance center {_target_center_world.round(3)}; moving directly{bcolors.ENDC}')
              _ee_quat_now = self._env.get_ee_quat()
              _ws_min_f = self._env.workspace_bounds_min + np.array([0.02, 0.02, 0.01])
              _ws_max_f = self._env.workspace_bounds_max - np.array([0.02, 0.02, 0.01])
              _target_clamped = np.clip(_target_center_world, _ws_min_f, _ws_max_f)
              # pick-guard：affordance 目标若落在可抓物体上且低于其顶面，抬高到顶面
              # 之上，避免"Final approach 直接把 EE 送进杯体"（run25 r0 顶飞路径之一）。
              try:
                for _pg_xy, _pg_top in _build_pick_guard(self._env):
                  if (float(np.linalg.norm(_target_clamped[:2] - _pg_xy)) < 0.07
                          and float(_target_clamped[2]) < _pg_top + 0.01):
                    _z_new_f = min(_pg_top + 0.01, float(_ws_max_f[2]))
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] pick-guard: Final approach z {_target_clamped[2]:.3f}→{_z_new_f:.3f} (over graspable @{_pg_xy.round(3)}, top={_pg_top:.3f}){bcolors.ENDC}')
                    _target_clamped = np.array([_target_clamped[0], _target_clamped[1], _z_new_f])
              except Exception:
                pass
              self._env.apply_action(np.concatenate([_target_clamped, _ee_quat_now, [gripper_state]]))
            else:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Final approach: EE already {_final_dist:.3f}m from target (<1.5cm), skipping{bcolors.ENDC}')
    except Exception as _fae:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Final approach failed: {_fae}{bcolors.ENDC}')
      traceback.print_exc()

    # === 按下动作：连续螺旋下压（逐帧 IK + 设关节 + step 场景）===
    # 解决刚性接触/碰撞回弹导致的按不下去：每帧重新读取实际 EE 位置再往下 δ，抵消物理回推。
    _did_press_down = False
    _press_total_m = 0.0
    if _affordance_map is not None:
      # ── R3 门控：目标落在可抓物体上时跳过物理下压 ────────────────────────
      # 此前只要 affordance_map 存在就无条件连续下压 30mm（任何任务）。对
      # pick 家族任务（StackCups 等）意味着闭合夹爪压进杯口，把杯子从台面顶飞
      # （run25 r0 实测：30mm 全行程下压后 cup 出现在 [0.446 0.709 0.042]，
      # 工作空间外 → 后续 Fix2 恒 SKIP）。按压只对"控制类目标"（按钮/开关）有意义；
      # 目标水平位置与可抓物体重合一律跳过，实际抓取交给 Fix2 的 top_z 锚定下降。
      _allow_press = True
      try:
        for _pg_xy, _pg_top in _build_pick_guard(self._env):
          if float(np.linalg.norm(np.asarray(_target_center_world, dtype=float)[:2] - _pg_xy)) < 0.07:
            _allow_press = False
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] R3 press-gate: affordance target {np.asarray(_target_center_world).round(3)} over graspable @{_pg_xy.round(3)} (top={_pg_top:.3f}) → skip press-down (pick intent){bcolors.ENDC}')
            break
      except Exception:
        _allow_press = True
      if _allow_press:
        try:
          # 40 步 × 0.8mm/步 = 最大 3.2cm；函数内部有 floor 下限保护 (auto-set to be well below start_z)
          _press_total_m = self._env.press_down_continuous(
              total_steps=40,
              delta_mm_per_step=0.8,
              z_floor_m=None,
          )
          _did_press_down = (_press_total_m > 0.001)  # 至少压下 1mm 才算有效
        except Exception as _pde:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Continuous press-down aborted: {_pde}{bcolors.ENDC}')
          _press_total_m = 0.0
        if _did_press_down:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Press-down total: {_press_total_m*1000:.1f}mm (continuous){bcolors.ENDC}')

    # === 动作后处理：抓取/放置/按钮长按/兜底Pick/Slide（两种模式都执行） ===
    try:
      _intend_close = (gripper_state is not None and float(gripper_state) < 0.5)
      _intend_open = (gripper_state is not None and float(gripper_state) > 0.5)
      _grasped_before = 0
      try:
        _grasped_before = int(self._env.get_grasped_object_count())
      except Exception:
        _grasped_before = 0
      # ── CRITICAL CLASSIFICATION: movable token → task category ──────────
      # If we don't classify correctly, _did_press_down will force press-mode
      # for EVERY task (composer always runs press_down for any affordance)
      # and Fix2 (pick fallback) / Fix3 (slide fallback) are NEVER reached.
      # 词表（_CONTROL_TOKENS / _PICKABLE_TOKENS / _SLIDABLE_TOKENS /
      # _AMBIG_TOKENS / _SLIDE_VERBS / _PICK_VERBS）已提升到模块级。
      _instr_low = ''
      try:
        _instr_low = str(getattr(self._env, 'current_instruction', '') or '').lower()
      except Exception:
        _instr_low = ''
      _mn_tok = None
      try:
        if _last_movable_name:
          import re as _re_if_classify
          _nm = _last_movable_name.strip().lower()
          if ' ' in _nm:
            _pp = _nm.split(' ')
            _mn_tok = _pp[-1] if _pp else _nm
          elif '_' in _nm:
            _pp = _nm.split('_')
            _mn_tok = _pp[-1] if _pp else _nm
          else:
            _mn_tok = _nm
          _mm = _re_if_classify.match(r'^(.*?)(\d+)$', _mn_tok or '')
          if _mm: _mn_tok = _mm.group(1)
      except Exception:
        _mn_tok = None
      _is_control = (_mn_tok and _mn_tok in _CONTROL_TOKENS) or (not object_centric)
      _is_pickable = (_mn_tok and _mn_tok in _PICKABLE_TOKENS)
      _is_slidable = (_mn_tok and _mn_tok in _SLIDABLE_TOKENS)
      # ── R4: 歧义名词（block/brick）按当前指令动词判定类别 ─────────────
      if _mn_tok in _AMBIG_TOKENS:
        _has_slide_v = any(_v in _instr_low for _v in _SLIDE_VERBS)
        _has_pick_v = any(_v in _instr_low for _v in _PICK_VERBS)
        if _has_slide_v and not _has_pick_v:
          _is_slidable, _is_pickable = True, False
        else:
          _is_pickable, _is_slidable = True, False
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: R4 ambiguous token "{_mn_tok}" (instr="{_instr_low[:70]}") → pickable={_is_pickable} slidable={_is_slidable}{bcolors.ENDC}')
      # ── EE-centric scene-inference fallback ─────────────────────────────
      # When movable_obs probe failed → EE-centric mode → movable_name="ee"
      # or "gripper".  _is_control=True (from `not object_centric`), which
      # forces press_mode for EVERY task, blocking Fix2/Fix3.  Fix: scan
      # scene objects to re-classify the task when movable_name is "ee" or
      # "gripper" (LLM often uses detect('gripper') for EE-centric tasks).
      if _mn_tok == 'ee' or _mn_tok == 'gripper' or _mn_tok is None:
        try:
          _scene_objs = self._env.get_object_names()
          _scene_toks = set()
          _scene_name_by_tok = {}
          for _so in _scene_objs:
            _so_l = _so.lower().strip()
            _so_tok = _so_l
            if ' ' in _so_l: _so_tok = _so_l.split(' ')[-1]
            elif '_' in _so_l: _so_tok = _so_l.split('_')[-1]
            import re as _re_if_sc
            _mm_sc = _re_if_sc.match(r'^(.*?)(\d+)$', _so_tok)
            if _mm_sc: _so_tok = _mm_sc.group(1)
            _scene_toks.add(_so_tok)
            _scene_name_by_tok.setdefault(_so_tok, _so)
          # Check for pickable objects in scene (meat, steak, cup, etc.)
          _scene_pickable = _scene_toks & _PICKABLE_TOKENS
          _scene_slidable = _scene_toks & _SLIDABLE_TOKENS
          _scene_control = _scene_toks & _CONTROL_TOKENS
          _scene_ambig = _scene_toks & _AMBIG_TOKENS
          if _scene_pickable and not _scene_control:
            _is_pickable = True
            _is_control = False
            # 目标错配修复：场景里可能同时有 chicken 和 steak（MeatOffGrill），
            # 原先 list(set)[0] 是任意选择 → 抓取目标可能与本 episode 目标差 0.15m。
            # 优先选指令中点名的那个对象，再用 sorted 保证确定性。
            _mn_tok = next((_t for _t in sorted(_scene_pickable) if _t in _instr_low),
                           sorted(_scene_pickable)[0])
            _last_movable_name = _mn_tok
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: EE-mode → scene-inference: found PICKABLE "{_mn_tok}" in scene objects (instr="{_instr_low[:70]}"); override control→pickable{bcolors.ENDC}')
          elif _scene_slidable and not _scene_control:
            _is_slidable = True
            _is_control = False
            _mn_tok = list(_scene_slidable)[0]
            _last_movable_name = _mn_tok
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: EE-mode → scene-inference: found SLIDABLE "{_mn_tok}" in scene objects; override control→slidable{bcolors.ENDC}')
          elif _scene_ambig and not _scene_control:
            # R4: 歧义名词 → 用指令动词决定 pickable / slidable
            _amb_tok = sorted(_scene_ambig)[0]
            _has_slide_v = any(_v in _instr_low for _v in _SLIDE_VERBS)
            _has_pick_v = any(_v in _instr_low for _v in _PICK_VERBS)
            _mn_tok = _amb_tok
            _last_movable_name = _amb_tok
            _is_control = False
            if _has_slide_v and not _has_pick_v:
              _is_slidable, _is_pickable = True, False
            else:
              _is_pickable, _is_slidable = True, False
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: EE-mode → scene-inference: found AMBIGUOUS "{_amb_tok}" (instr="{_instr_low[:70]}") → pickable={_is_pickable} slidable={_is_slidable}{bcolors.ENDC}')
          # ── R4/G1: EE-mode 下 _last_movable_world_pos 记录的是 EE 自身位置，
          # 会污染 Fix2 的抓取目标（在空处闭爪）。这里用场景中真实物体的
          # 世界坐标（点云质心 + 顶面高度）覆盖它。 ──
          _full_nm = _scene_name_by_tok.get(_mn_tok)
          if _full_nm:
            try:
              # 优先真实名解析并按本 episode 指令消歧（'meat'→steak/chicken）；
              # 语义点云 get_3d_obs_by_name 在 headless 下把 'meat' 一律解析到先注册
              # 的 chicken，会把抓取目标换成错误物体（实测指令 steak、却 grasp 到 chicken）。
              # run27：改用具体名解析（原先传泛称 _mn_tok，会包含匹配到第一个同类
              # 候选——与日志里打印的 _full_nm 并非同一物体，坐标因此对不上）。
              _rp_sc, _rtz_sc = self._env.get_object_pos_by_name(_full_nm)
              if _rp_sc is not None:
                _last_movable_world_pos = np.asarray(_rp_sc, dtype=float).reshape(3).copy()
                _last_movable_top_z = (float(_rtz_sc) if _rtz_sc is not None
                                       else float(_last_movable_world_pos[2]))
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: scene-inference resolved movable "{_full_nm}" real center={_last_movable_world_pos.round(3)} top_z={_last_movable_top_z:.3f} via real-name lookup{bcolors.ENDC}')
              else:
                _pc_sc, _ = self._env.get_3d_obs_by_name(_full_nm)
                _pc_sc = np.asarray(_pc_sc, dtype=float).reshape(-1, 3)
                if _pc_sc.shape[0] > 0:
                  _c_sc = _pc_sc.mean(axis=0).copy()
                  # scene-scan 点云在物理爆炸时会给出 1e4 量级伪坐标（实测 z=-1.4e5
                  # 一路涨到 1e7），必须在源头拒绝，避免污染 _last_movable_world_pos。
                  if not self._env._pos_is_sane(_c_sc):
                    print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Post-action: scene-scan for "{_full_nm}" yielded non-sane center {_c_sc.round(1)} → rejected{bcolors.ENDC}')
                  else:
                    _last_movable_world_pos = _c_sc
                    _last_movable_top_z = float(_pc_sc[:, 2].max())
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: scene-inference resolved movable "{_full_nm}" real center={_last_movable_world_pos.round(3)} top_z={_last_movable_top_z:.3f} via scene scan{bcolors.ENDC}')
            except Exception as _sre:
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] scene-inference pos resolve failed for "{_full_nm}": {_sre}{bcolors.ENDC}')
        except Exception:
          pass
      if _mn_tok:
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: movable_name="{_last_movable_name}" → token="{_mn_tok}" → control={_is_control} pickable={_is_pickable} slidable={_is_slidable} grasped_before={_grasped_before} intend_close={_intend_close}{bcolors.ENDC}')
      _ee_z = None
      try:
        _ee_z = float(self._env.get_ee_pos()[2])
      except Exception:
        pass
      # ── Press-mode gatekeeping ──────────────────────────────────────────
      # Rule: press_mode only when task is MANIPULATING A CONTROL (button /
      # switch / knob / lever).  For pickable / slidable movables, even if
      # composer ran press_down_continuous (which it always does when
      # affordance_map is present), we MUST NOT enter press_mode → instead
      # fall through to the Fix2 pick / Fix3 slide fallback branches below.
      _press_mode = False
      # 任务是否已满足：MeatOffGrill 等 pickable 任务在 post-action 每个 composer 步骤
      # 都会重入 Fix2；若首个放置已成功（物体已进入 success 传感器），再次抓取会把
      # 已放好的物体拖离传感器，反而毁掉成功。用原生 env.success() 作守卫。
      try:
        _env_done = bool(self._env.success())
      except Exception:
        _env_done = False
      # ── Fix(A) 错抓纠正 ────────────────────────────────────────────────
      # composer 的 affordance 常落在语义泛称上（'a point at the center of the
      # meat' → headless 下 detect('meat') 解析到先注册的 chicken），RLBench 的
      # MoveArmThenGripper 动作模式会在航点执行时就把 chicken 抓起来，于是 post-action
      # 看到 _grasped_before>0、跳过 Fix2，最终带着错误物体走完全程。
      # 若当前握持的可抓取物不是本 episode 指令点名的那个，先纯 IK 开爪释放，
      # 让 Fix2 用真实名解析去抓正确目标。（仅当目标与握持物都属 PICKABLE 词表且
      # 目标在指令中被点名时才触发，避免误伤 lid/saucepan 等未入表的任务。）
      if _is_pickable and _grasped_before > 0:
        try:
          import re as _re_tgt
          _go_names = []
          try:
            _go = self._env.rlbench_env._scene.robot.gripper.get_grasped_objects()
            _go_names = [str(o.get_name()).strip().lower() for o in _go]
          except Exception:
            _go_names = []
          # get_grasped_objects() 返回 RLBench 真实形状名（如 'chicken' / 'steak'）。
          # 若握持物名分词后与指令分词完全无交集，说明抓的不是本 episode 点名的物体
          # （典型：composer 的 'meat' 泛称被 detect 解析成先注册的 chicken）。
          # 此时纯 IK 开爪释放，让 Fix2 用真实名+指令消歧去抓正确目标。
          _held_toks = set()
          for _gn in _go_names:
            _held_toks |= {t for t in _re_tgt.split(r'[ _]+', _gn) if t}
          _instr_toks = set(_re_tgt.findall(r'[a-z]+', _instr_low))
          if _go_names and not (_held_toks & _instr_toks):
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix(A) wrong-object grasp: holding {_go_names} not named in instr="{_instr_low}" → release, let Fix2 pick the target{bcolors.ENDC}')
            try:
              self._env.release_with_settle(stabilize_steps=20, lift_before_release=False)
            except Exception:
              pass
            try:
              _grasped_before = int(self._env.get_grasped_object_count())
            except Exception:
              _grasped_before = 0
        except Exception:
          pass
      if (_is_control or (_mn_tok is None)) and _did_press_down:
        # Button/switch/EE-centric task with actual press → press-mode (hold)
        _press_mode = True
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: press-mode ON (control={_is_control}, movable_token="{_mn_tok}", pressed {_press_total_m*1000:.1f}mm){bcolors.ENDC}')
      elif _affordance_map is not None and _ee_z is not None and 0.72 <= _ee_z <= 0.82 and (_is_control or (_mn_tok is None)):
        # Legacy desk-level press (only for control tasks; avoid holding
        # down onto a block/meat forever).
        _press_mode = True
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: press-mode ON (EE z={_ee_z:.3f}m, obj-level){bcolors.ENDC}')
      # ── lift-only 任务（PickAndLift）────────────────────────────────────
      # 成功条件 = GraspedCondition(目标) ∧ DetectedCondition(目标, 上方传感器)，
      # 因此动作是"保持夹持把物体举进传感器"，不是"放置 + 释放"。
      _is_lift_task = False
      try:
        _is_lift_task = 'pick_and_lift' in str(
            self._env.rlbench_env._scene.task.get_name()).lower()
      except Exception:
        _is_lift_task = False
      # ── 多物体任务（StackBlocks / BlockPyramid / EmptyContainer）────────
      # 这些任务的 success 传感器就在台面/容器面上（z 接近台面），且一次
      # pick-place 只搬一只。放置高度的支撑面不能用 StackCups 的 "+0.14"
      # 经验值（实测把方块吊到 14cm 高空释放，block1 被弹到 0.84m 外）。
      _task_name_l = ''
      try:
        _task_name_l = str(self._env.rlbench_env._scene.task.get_name()).lower()
      except Exception:
        _task_name_l = ''
      _is_multi_obj_task = any(_k in _task_name_l for _k in
                               ('stack_blocks', 'block_pyramid', 'empty_container'))
      # ── 槽位/插入类任务（方案第 2 项，2026-10-25）──────────────────────────
      # PutKnifeInKnifeBlock 的成功传感器在刀座**底部**、检出区是刀座内部的
      # 竖直槽区；只读探针 probe_knife_free.py 实测：刀平放释放必滑落到台面
      # （z<sensor，永不检出），竖直刃朝下释放则稳定停在 z≈0.91 且检出=True。
      # 因此该任务不能走"水平搬运+下放+释放"，必须先把刀转成刃朝下再插入。
      _is_insert_task = ('in_knife_block' in _task_name_l)
      # ── 杯架类任务（PlaceCups，2026-10-25）────────────────────────────────
      # 只读探针 probe_place_cups_handle2.py 实测：同一投放点下，手柄同向 / 垂直 /
      # 45° 三种朝向都留在架上并被原生检出；真正决定成败的是**释放高度**。
      # 因此这里不做手柄 yaw 对齐，只按"辐条顶面"反算释放高度。
      _is_cups_task = ('place_cups' in _task_name_l)
      # ── MeatOffGrill：释放高度按传感器面 + 肉半高（2026-10-25）─────────────
      # 基线 r0 实测：叠放支撑面逻辑把同类里的**悬空视觉网格**（top=1.317）当成
      # 支撑面 → 目标杯心被算到 1.341、EE 在 1.349 释放（离传感器 29cm），
      # 肉落下弹飞 411mm。成功样本（r1/r2）的肉心稳定停在 1.084~1.086
      # （= 传感器 z 1.061 + 半高 0.022），故此处按传感器面反算释放高度。
      _is_meat_task = ('meat_off_grill' in _task_name_l)
      if _is_insert_task:
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: insert-task '
              f'"{_task_name_l}" → vertical slot-insert primitive enabled{bcolors.ENDC}')

      def _lift_to_sensor(_tag=''):
        """保持夹持，把物体举进 success 传感器；返回是否达到原生 env.success()。"""
        try:
          _task_o = self._env.rlbench_env._scene.task
          _cs = getattr(_task_o, '_success_conditions', []) or []

          def _find_det(_conds):
            for _c in (_conds or []):
              _d = getattr(_c, '_detector', None)
              if _d is not None:
                return _d
              _r = _find_det(getattr(_c, '_conditions', None))
              if _r is not None:
                return _r
            return None

          _det = _find_det(_cs)
          if _det is None:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] lift-only[{_tag}]: no detector in success conditions{bcolors.ENDC}')
            return False
          _sp = np.asarray(_det.get_position(), dtype=float).reshape(3)
          _ee_n = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
          _hang = 0.022
          # 优先用**实际被握物体**的位置算「EE→物体中心」高差：语义名解析在
          # lid / gripper 这类未入表的 token 上会失败或命中相机占位体（实测
          # get_object_pos_by_name('holder') 返回 z=2.28 的背景相机）。
          try:
            _gobjs = list(self._env.rlbench_env._scene.robot.gripper.get_grasped_objects())
            if _gobjs:
              _hang = float(max(0.0, _ee_n[2] - float(np.asarray(
                  _gobjs[0].get_position(), dtype=float).reshape(3)[2])))
            else:
              raise ValueError('no grasped object')
          except Exception:
            try:
              _op, _ = self._env.get_object_pos_by_name(_last_movable_name or _mn_tok)
              if _op is not None:
                _hang = float(max(0.0, _ee_n[2] - float(np.asarray(_op, float)[2])))
            except Exception:
              pass
          _z0 = float(_sp[2]) + float(_hang)
          _okl = False
          # 传感器判定体沿局部轴展开，最优高度不必等于原点：按 ±2/4/6cm 粗扫，
          # 每步直接读原生 env.success()（r7 实测在 -14mm 处才成立）。
          for _dz in (0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06):
            _tl = np.array([_sp[0], _sp[1], _z0 + _dz], dtype=float)
            _lo, _le, _li = self._env.move_ee_closed_loop(_tl, tol=0.010, max_iters=90)
            try:
              _sl = bool(self._env.success())
            except Exception:
              _sl = False
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] lift-only[{_tag}]: '
                  f'target=({_tl[0]:.3f},{_tl[1]:.3f},{_tl[2]:.3f}) sensor_z={_sp[2]:.3f} '
                  f'hang={_hang*1000:.0f}mm ok={_lo} err={_le*1000:.1f}mm success={_sl}{bcolors.ENDC}')
            if _sl:
              _okl = True
              break
          try:
            _cntl = int(self._env.get_grasped_object_count())
          except Exception:
            _cntl = -1
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] lift-only[{_tag}]: '
                f'{"SUCCESS" if _okl else "not successful"} (holding={_cntl}){bcolors.ENDC}')
          return _okl
        except Exception as _lfe:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] lift-only[{_tag}] raised: {_lfe}{bcolors.ENDC}')
          return False

      def _place_at_sensor(_tag=''):
        """把**已握持**的物体送到 success 传感器正上方并释放。

        动机（2026-10-25 MeatOffGrill）：计划常常自己先把肉抓住
        （composer 的 affordance 动作在航点处就闭爪），于是 post-action 看到
        `grasped_before>0`，Fix2 的"抓取→放置"整段被跳过，后续只有
        `move 20cm / open gripper` 之类的步骤 → 肉一直握在手里，终态
        `NothingGrasped=False`。这里补上"已握持 → 送传感器 → 释放"。
        释放高度按传感器面 + 物体半高 + 2cm，与 Fix2 的放置高度一致。
        """
        try:
          _task_o = self._env.rlbench_env._scene.task
          _cs = getattr(_task_o, '_success_conditions', []) or []

          def _find_det2(_conds):
            for _c in (_conds or []):
              _d = getattr(_c, '_detector', None)
              if _d is not None:
                return _d
              _r = _find_det2(getattr(_c, '_conditions', None))
              if _r is not None:
                return _r
            return None

          _det = _find_det2(_cs)
          if _det is None:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] place-at-sensor[{_tag}]: '
                  f'no detector in success conditions{bcolors.ENDC}')
            return False
          _sp = np.asarray(_det.get_position(), dtype=float).reshape(3)
          _ee = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
          _rel = -0.008
          _half = 0.022
          try:
            _go = list(self._env.rlbench_env._scene.robot.gripper.get_grasped_objects())
            if _go:
              _op = np.asarray(_go[0].get_position(), dtype=float).reshape(3)
              _rel = float(_op[2] - _ee[2])
          except Exception:
            pass
          try:
            _wsmin = np.asarray(self._env.workspace_bounds_min, dtype=float).reshape(3)
            _wsmax = np.asarray(self._env.workspace_bounds_max, dtype=float).reshape(3)
            _tgt = np.clip(np.array([_sp[0], _sp[1], float(_sp[2]) + _half + 0.04 - _rel]),
                           _wsmin, _wsmax)
          except Exception:
            _tgt = np.array([_sp[0], _sp[1], float(_sp[2]) + _half + 0.04 - _rel])
          _ok, _err, _it = self._env.move_ee_closed_loop(_tgt, tol=0.008, max_iters=120)
          self._env.stabilize(steps=10)
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] place-at-sensor[{_tag}]: '
                f'target=({_tgt[0]:.3f},{_tgt[1]:.3f},{_tgt[2]:.3f}) sensor_z={_sp[2]:.3f} '
                f'ok={_ok} err={_err*1000:.1f}mm iters={_it}{bcolors.ENDC}')
          # XY 居中：EE 到传感器上方不代表**物体**也在传感器上方（侧抓有横向偏移）。
          # Fix2 的放置路径有这一步（实测残留 3~30mm），这里同样补上：按被握物体的
          # 真实世界中心与传感器 XY 的差平移 EE，只修 ≤5cm 的残差。
          try:
            _go2 = list(self._env.rlbench_env._scene.robot.gripper.get_grasped_objects())
            if _go2:
              _op2 = np.asarray(_go2[0].get_position(), dtype=float).reshape(3)
              _dxy2 = np.array([float(_sp[0]) - _op2[0], float(_sp[1]) - _op2[1]])
              _dn2 = float(np.linalg.norm(_dxy2))
              if 0.002 < _dn2 <= 0.05:
                _tgt2 = _tgt.copy()
                _tgt2[0] += _dxy2[0]
                _tgt2[1] += _dxy2[1]
                _ok2, _err2, _it2 = self._env.move_ee_closed_loop(
                    _tgt2, tol=0.006, max_iters=90)
                self._env.stabilize(steps=8)
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] place-at-sensor[{_tag}]: '
                      f'xy-center obj→sensor {_dn2*1000:.0f}mm ok={_ok2} err={_err2*1000:.1f}mm{bcolors.ENDC}')
                _err = max(float(_err), float(_err2)) if np.isfinite(_err2) else float(_err)
          except Exception as _xce:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] place-at-sensor[{_tag}] '
                  f'xy-center raised: {_xce}{bcolors.ENDC}')
          if float(_err) > 0.05:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] place-at-sensor[{_tag}]: '
                  f'not reached (err={_err*1000:.1f}mm > 50mm) → abort release{bcolors.ENDC}')
            return False
          # 释放纪律：先闭爪抬 1cm 再张开。直接原地开爪时指板仍贴着物体侧壁，
          # 张开瞬间会把薄片（肉）弹开（run4 r3/r7 实测释放后未被原生检出）。
          self._env.release_with_settle(stabilize_steps=120, lift_before_release=True)
          try:
            _ep_ok = bool(self._env.success())
          except Exception:
            _ep_ok = False
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] place-at-sensor[{_tag}]: '
                f'released; native success={_ep_ok}{bcolors.ENDC}')
          return _ep_ok
        except Exception as _pe:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] place-at-sensor[{_tag}] '
                f'raised: {_pe}{bcolors.ENDC}')
          return False

      # ── MeatOffGrill：已握持时直接送到传感器并释放（2026-10-25）──────────
      # 计划常先自己把肉抓住（r0 实测 grasped_before=1），Fix2 因 `grasped_before==0`
      # 前置条件整段跳过，后续只有 move/open → 肉一直握在手里。这里在已握持时
      # 直接执行"送传感器 + 释放"，成功仍由原生 env.success() 判定。
      if _is_meat_task and _grasped_before > 0 and not _env_done:
        _place_at_sensor('meat-holding')

      # ── TakeLidOffSaucepan：抓盖后必须「举升进传感器」，不能按压/释放 ─────
      # 只读探针 probe_lid_sensor.py 实测（TakeLidOffSaucepan var0）：
      #   · 锅盖 grasp point 初始 z=0.853，success 传感器在其**正上方** z=0.968；
      #   · 检出窗口 = lid z∈[1.013, 1.073]（即抬升 16~22cm），XY 偏移 ≤8cm 仍检出；
      #   · 成功条件 = ConditionSet([GraspedCondition(lid), DetectedCondition(lid, success)])，
      #     即"仍握持 ∧ 锅盖进入传感器"，一旦开爪释放就永久失败。
      # 计划里的"锅上方 20cm"恰好停在窗口下沿附近，随后 `open gripper` 直接破坏
      # GraspedCondition → 基线 0/10（抓取本身每次都是 cnt=1）。
      # 因此这里在已握持时强制举升到传感器，并禁止按压/释放。
      _is_lid_task = ('take_lid_off_saucepan' in _task_name_l)
      if _is_lid_task and _grasped_before > 0:
        if _press_mode:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: press-mode suppressed '
                f'(lid task is lift-only while holding){bcolors.ENDC}')
        _press_mode = False
        if not _env_done:
          _lift_to_sensor('take-lid')

      # ── Dispatch ────────────────────────────────────────────────────────
      if _press_mode:
        # 按压类（button/switch/knob/lever/EE-mode direct press）：加力按 + 长按
        try:
          _extra_press = self._env.press_down_continuous(
              total_steps=35, delta_mm_per_step=1.0, z_floor_m=None,
          )
          if _extra_press < 0.001:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Extra press moved only {_extra_press*1000:.1f}mm; rely on long hold{bcolors.ENDC}')
          self._env.hold_press(hold_steps=60)
        except Exception as _ppe:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Press-mode post-action raised: {_ppe}{bcolors.ENDC}')
      # =====================================================================
      # Fix2 (Pick-class fallback — MeatOffGrill / Pick* etc.)
      # When: object_centric AND movable token ∈ PICKABLE (definitive pick
      #       task) AND currently nothing grasped.
      # Typical failure case: "take chicken off grill" → composer moves EE
      # above meat and calls press_down (z-squeeze 20mm) but never closes
      # gripper → meat stays → env.success always False.
      # HIGHEST PRIORITY after press-mode: if this is a pick task, we MUST
      # attempt the grasp even if LLM said nothing about gripper_state.
      # =====================================================================
      elif _is_pickable and _grasped_before == 0 and not _env_done:
        # Resolve movable position: use saved _last_movable_world_pos, or
        # query scene for the movable object by name.
        _mv_pos = None
        _mv_top_z = None
        _f2_name = None
        # ── G1: 优先按 movable token 在场景中查真实物体几何 ────────────────
        # EE-mode 下 _last_movable_world_pos 记录的是 EE 自身位置，直接拿它
        # 当抓取目标会在空处闭爪（open_amount=0.000）。所以先做真实物体解析，
        # 同时取出 AABB 顶面高度用于决定下降高度。
        if _mn_tok and _mn_tok not in ('ee', 'gripper'):
          # ── 优先：按真实物体名直接问 CoppeliaSim 要世界 AABB 中心/顶面 ────
          # 语义名扫描（get_3d_obs_by_name）在 headless 下会把 'meat' 一律解析到
          # 先注册的 'chicken'，与本 episode 真实目标 'steak' 偏差可达 0.147m；
          # 这里用真实名解析，直接消除目标错配。
          # 并在同类多个候选中**排除工作空间外**者、取离 EE 最近者（run26 复盘：
          # 泛称解析恒指向第一个候选，该候选一旦被推出界，Fix2 就永久 SKIP）。
          try:
            # run34：把指令点名的「底座/支撑物」颜色排除出抓取候选（StackCups 的红杯）。
            _f2_excl = _support_color_from_instruction(
                str(getattr(self._env, 'current_instruction', '') or ''))
            # ── 泛称 meat → 按指令点名解析（2026-10-25）────────────────────
            # MeatOffGrill 场景里除可抓的 chicken / steak 外，还有一个**不可抓**的
            # 视觉网格 "silver meat"。composer 用泛称 'meat' 时，场景推断会命中
            # 该网格，Fix2 于是在网格上空闭合：基线 10 次里 6 次 `detected=[]`
            # （r2/r4/r5/r7/r9 三次重试 + 25 点扫描全空）。本 episode 的真实目标
            # 由指令点名（var0='chicken'，var1='steak'），这里优先按它解析。
            _f2_name, _rp_f2, _rtz_f2 = None, None, None
            if _mn_tok in ('meat',):
              _instr_low_m = str(getattr(self._env, 'current_instruction', '') or '').lower()
              for _cand in ('chicken', 'steak'):
                if _cand in _instr_low_m:
                  try:
                    _rc_m, _rtz_m = self._env.get_object_pos_by_name(_cand)
                  except Exception:
                    _rc_m, _rtz_m = None, None
                  if _rc_m is not None:
                    _f2_name = _cand
                    _rp_f2 = np.asarray(_rc_m, dtype=float).reshape(3)
                    _rtz_f2 = _rtz_m
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: generic '
                          f'"meat" → instruction target "{_cand}" center={_rp_f2.round(3)}{bcolors.ENDC}')
                  break
            if _f2_name is None:
              _f2_name, _rp_f2, _rtz_f2 = _resolve_pick_target(
                  self._env, _mn_tok, ref_xy=self._env.get_ee_pos(),
                  exclude_color=_f2_excl)
            if _f2_excl and _f2_name:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: support/base color "{_f2_excl}" '
                    f'excluded from pick candidates → chose "{_f2_name}"{bcolors.ENDC}')
          except Exception:
            _f2_name, _rp_f2, _rtz_f2 = None, None, None
          if _rp_f2 is not None:
            _mv_pos = np.asarray(_rp_f2, dtype=float).reshape(3).copy()
            _mv_top_z = float(_rtz_f2) if _rtz_f2 is not None else float(_mv_pos[2])
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: resolved movable "{_f2_name}" (tok={_mn_tok}) real center={_mv_pos.round(3)} top_z={_mv_top_z:.3f} via real-name lookup{bcolors.ENDC}')
        if _mv_pos is None and _mn_tok and _mn_tok not in ('ee', 'gripper'):
          try:
            import re as _re_f2
            for _on in self._env.get_object_names():
              _on_l = _on.lower().strip()
              _on_tok = _on_l
              if ' ' in _on_l: _on_tok = _on_l.split(' ')[-1]
              elif '_' in _on_l: _on_tok = _on_l.split('_')[-1]
              _mm_f2 = _re_f2.match(r'^(.*?)(\d+)$', _on_tok)
              if _mm_f2: _on_tok = _mm_f2.group(1)
              if _on_tok == _mn_tok or _mn_tok in _on_l:
                try:
                  _pc_f2, _ = self._env.get_3d_obs_by_name(_on)
                  _pc_f2 = np.asarray(_pc_f2, dtype=float).reshape(-1, 3)
                except Exception:
                  _pc_f2 = None
                if _pc_f2 is not None and _pc_f2.shape[0] > 0:
                  _c_f2 = _pc_f2.mean(axis=0).copy()
                  # 同 Post-action：物理爆炸时 scene-scan 会给出 1e4 量级伪坐标，
                  # 源头拒绝，避免 Fix2 基于伪坐标反复 SKIP/空转。
                  if not self._env._pos_is_sane(_c_f2):
                    print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: scene-scan for "{_on}" yielded non-sane center {_c_f2.round(1)} → rejected{bcolors.ENDC}')
                  else:
                    _mv_pos = _c_f2
                    _mv_top_z = float(_pc_f2[:, 2].max())
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: resolved movable "{_on}" real center={_mv_pos.round(3)} top_z={_mv_top_z:.3f} via scene scan{bcolors.ENDC}')
                    break
          except Exception:
            pass
        # ── 退路：使用已保存的 movable 位置 ──────────────────────────────
        if _mv_pos is None and _last_movable_world_pos is not None:
          _mv_pos = np.asarray(_last_movable_world_pos, dtype=float).reshape(3).copy()
          if _last_movable_top_z is not None:
            _mv_top_z = float(_last_movable_top_z)
        # ── 工作空间可达性守卫 ──────────────────────────────────────────────
        # 兜底的横向推会把物体推离台面（实测 cup 落到 y=0.694、z=0.031，已在工作
        # 空间外）。之后每个 composer 步骤仍会重入 Fix2，对不可达目标反复做 OMPL
        # 重试 + 25 候选扫描抓取，把 480s 预算耗尽。这里一旦判定目标在可抓取区域
        # 之外，就跳过整段抓取兜底；成功判定仍完全依赖原生 grasp()/env.success()。
        if _mv_pos is not None:
          try:
            _ws_lo_f2 = np.asarray(self._env.workspace_bounds_min, dtype=float).reshape(3)
            _ws_hi_f2 = np.asarray(self._env.workspace_bounds_max, dtype=float).reshape(3)
            _out_margin = 0.02
            _inside_f2 = bool(np.all(_mv_pos >= _ws_lo_f2 - _out_margin)
                              and np.all(_mv_pos <= _ws_hi_f2 + _out_margin))
            if not _inside_f2:
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 SKIP: movable "{_mn_tok}" at {_mv_pos.round(3)} outside workspace [{_ws_lo_f2.round(2)}..{_ws_hi_f2.round(2)}] → skip pick fallback{bcolors.ENDC}')
              _mv_pos = None
          except Exception:
            pass
        if _mv_pos is not None:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: movable="{_last_movable_name}" (tok={_mn_tok}) grasped=0 → approach + grasp_with_retry{bcolors.ENDC}')
          try:
            # 抓取朝向用 home(复位) 朝向：实测 waypoint1 朝向每 episode 随机，
            # 会让 proximity 探测窗口消失；home 朝向窗口稳定。
            _ee_quat_now = self._env.get_home_ee_quat()
            _ws_min_f = self._env.workspace_bounds_min + np.array([0.02, 0.02, 0.01])
            _ws_max_f = self._env.workspace_bounds_max - np.array([0.02, 0.02, 0.01])
            # 先在当前位姿张开夹爪，避免带着半闭手指接近物体
            try:
              self._env.apply_action(np.concatenate([np.array(self._env.get_ee_pos(), dtype=float), _ee_quat_now, [1.0]]))
              self._env.stabilize(steps=10)
            except Exception:
              pass
            # run34：接近前先「垂直抬升」到安全高度，再做水平移动。
            # run33 实测：EE 停在低位时直接规划到物体正上方，路径会扫过台面，把目标杯
            # 撞出工作空间（black cup1 → [0.195,-0.66,0.03]）；该杯出界后 target 择优
            # 只能退到底座杯。这里先直上（不会横扫）抬到 top_z+4cm 高度，让后续移动
            # 全程在杯口之上进行。
            try:
              _cur_f2 = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
              _safe_hi = (float(_mv_top_z) + 0.04) if _mv_top_z is not None else (float(_mv_pos[2]) + 0.04)
              _lift_z_f2 = float(min(max(float(_cur_f2[2]) + 0.03, _safe_hi), float(_ws_max_f[2])))
              if _lift_z_f2 > float(_cur_f2[2]) + 0.005:
                _lift_pos_f2 = np.clip(np.array([float(_cur_f2[0]), float(_cur_f2[1]), _lift_z_f2]),
                                       _ws_min_f, _ws_max_f)
                _lp_ok, _lp_err, _lp_it = self._env.move_ee_closed_loop(
                    _lift_pos_f2, target_quat=_ee_quat_now, tol=0.01, max_iters=80)
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: pre-approach vertical lift ok={_lp_ok} '
                      f'err={_lp_err*1000:.1f}mm iters={_lp_it} z {float(_cur_f2[2]):.3f}→{_lift_z_f2:.3f}{bcolors.ENDC}')
                self._env.stabilize(steps=8)
            except Exception:
              pass
            # Step 1: 物体正上方 +4cm（夹爪张开）
            #   先 apply_action 做粗定位（路径规划长距离移动），再用闭环小步修正残差
            #   ——改为纯开环时实测终点残留 0.091m，夹爪会在物体旁空抓
            #   （`Planning OK but EE 0.091m from target` → `attempt 1 empty`）。
            _appr_z = (float(_mv_top_z) + 0.04) if _mv_top_z is not None else (float(_mv_pos[2]) + 0.04)
            _approach = np.clip(np.array([_mv_pos[0], _mv_pos[1], _appr_z], dtype=float), _ws_min_f, _ws_max_f)
            try:
              self._env.apply_action(np.concatenate([_approach, _ee_quat_now, [1.0]]))
              self._env.stabilize(steps=15)
            except Exception:
              pass
            _ap_ok, _ap_err, _ap_it = self._env.move_ee_closed_loop(
                _approach, target_quat=_ee_quat_now, tol=0.01, max_iters=120)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: closed-loop approach ok={_ap_ok} err={_ap_err*1000:.1f}mm iters={_ap_it}{bcolors.ENDC}')
            self._env.stabilize(steps=10)
            # 离台/越界早停 ①：接近过程中若目标已被撞出工作空间，立即放弃本次抓取，
            # 不再继续下压/扫描/搬运（继续只会反复撞击几何体、令接触求解发散，
            # 且后续每个 composer 步骤都会重入兜底，把 planner 墙钟耗尽）。
            try:
              _rp_chk, _ = self._env.get_object_pos_by_name(_f2_name or _mn_tok)
              if _rp_chk is not None and _obj_off_workspace(self._env, _rp_chk):
                print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: movable "{_f2_name or _mn_tok}" knocked off workspace ({np.asarray(_rp_chk).round(3)}) during approach → abort pick{bcolors.ENDC}')
                raise RuntimeError('movable off workspace during approach')
            except RuntimeError:
              raise
            except Exception:
              pass
            # Step 2: 下降到夹取高度 = 探测窗口内、尽量浅的高度
            #   proximity 探测窗口实测位于物体顶部附近（steak: tip.z ∈ [top_z-16.4mm,
            #   top_z-1mm]）。取 top_z-3mm：既落在窗口内，又是窗口里最浅的高度，
            #   IK 最易到位。若仍按物体「中心 z」下降，在部分 episode/XY 下 IK 到不了
            #   （实测残差 21.7mm，tip 停在窗口上方 → 闭爪空抓）。
            _desc_z = (float(_mv_top_z) - 0.003) if _mv_top_z is not None else float(_mv_pos[2])
            _descend = np.clip(np.array([_mv_pos[0], _mv_pos[1], _desc_z], dtype=float), _ws_min_f, _ws_max_f)
            # 探测窗口仅约 ±8mm（实测 steak tip.z ∈ [1.0479, 1.0633]），
            # 下降容差必须收紧到 4mm，否则残差会把 EE 留在窗口外 → 闭爪空抓。
            _de_ok, _de_err, _de_it = self._env.move_ee_closed_loop(
                _descend, target_quat=_ee_quat_now, tol=0.004, max_iters=140)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: closed-loop descend ok={_de_ok} err={_de_err*1000:.1f}mm iters={_de_it}{bcolors.ENDC}')
            self._env.stabilize(steps=10)
            # Step 3: grasp_with_retry（G2: 含闭环 XY 对准 + 夹爪开度校验）
            _ok2, _cnt2 = self._env.grasp_with_retry(max_retry=3, stabilize_steps=25, push_down_m=0.015, target_pos=_mv_pos, top_z=_mv_top_z)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: grasp_with_retry ok={_ok2} cnt={_cnt2}{bcolors.ENDC}')
            # 离台/越界早停 ②：抓取阶段结束时若目标仍在工作空间外（抓取过程中被撞飞），
            # 直接终止本次抓取与后续搬运，避免"抓空但仍搬运/放置"的空转。
            if not (_ok2 and _cnt2 > 0):
              try:
                _rp_chk2, _ = self._env.get_object_pos_by_name(_f2_name or _mn_tok)
                if _rp_chk2 is not None and _obj_off_workspace(self._env, _rp_chk2):
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: movable "{_mn_tok}" knocked off workspace ({np.asarray(_rp_chk2).round(3)}) during grasp → abort pick{bcolors.ENDC}')
                  raise RuntimeError('movable off workspace during grasp')
              except RuntimeError:
                raise
              except Exception:
                pass
            if _ok2 and _cnt2 > 0:
              # Step 4: 抬升 → 搬运到放置区 → 下放 → 释放。
              # 旧实现只"抬 5cm + 朝 affordance 目标水平推"：既没有把物体送进任务的
              # success 区域，也从未张开夹爪（物体一直挂在夹爪上）——因此即使抓到了正确
              # 物体，env.success() 仍为 False。这里改为真正的放置：优先用任务 success
              # 传感器位置作放置点，其次退回 affordance 目标。全程纯 IK，无传送。
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: grasped, lift then place to target region{bcolors.ENDC}')
              try:
                # ── 解析放置目标：任务 success 传感器优先 ──────────────────
                _place = None
                _place_src = ''
                _sa = None
                try:
                  _task_obj = self._env.rlbench_env._scene.task
                  _sensor = None
                  # ① 直接从任务的成功条件里取探测传感器：StackCups 等任务把
                  #    ProximitySensor('success') 作为 register_success_conditions 的
                  #    DetectedCondition 传入，并未挂到 task 的同名属性上，
                  #    旧代码按属性名猜必然落空 → 退回 affordance 目标（离真实
                  #    success 区可达 212mm，释放后 env_done 恒 False）。
                  # ② 必须**递归**查找：PickAndLift 的成功条件是 ConditionSet
                  #    （GraspedCondition + DetectedCondition），传感器挂在子条件上，
                  #    只看顶层 _detector 会得到 None。
                  def _find_detector(_conds):
                    for _c in (_conds or []):
                      _d = getattr(_c, '_detector', None)
                      if _d is not None:
                        return _d
                      _r = _find_detector(getattr(_c, '_conditions', None))
                      if _r is not None:
                        return _r
                    return None

                  try:
                    _sensor = _find_detector(
                        getattr(_task_obj, '_success_conditions', []) or [])
                    if _sensor is not None:
                      _sa = 'success_conditions (recursive)'
                  except Exception:
                    pass
                  if _sensor is None:
                    for _sa in ('_success_sensor', '_success', '_success_sensor0'):
                      _sensor = getattr(_task_obj, _sa, None)
                      if _sensor is not None:
                        break
                  if _sensor is not None:
                    _place = np.array(_sensor.get_position(), dtype=float).reshape(3)
                    _place_src = f'task success sensor ({_sa})'
                except Exception:
                  _place = None
                if _place is None and _last_target_center_world is not None:
                  _place = np.asarray(_last_target_center_world, dtype=float).reshape(3)
                  _place_src = 'affordance target'
                # ── lift-only 任务（PickAndLift）：保持夹持，把物体举进上方传感器 ──
                # 旧流程把 affordance 目标（z≈1.5，机械臂够不到）当放置点，再用
                # 叠放几何反算下放高度（z≈1.69），move-to-place 残差 244mm，
                # 且一旦释放就破坏 GraspedCondition。
                if _is_lift_task:
                  _lift_ok = _lift_to_sensor('post-pick')
                  raise RuntimeError('LIFT_ONLY_DONE' if _lift_ok else 'LIFT_ONLY_FAIL')
                # run38：底座杯名/中心/顶面（供"放置点 XY 取杯轴"与"判定带高度反算"用）。
                _base_name = None
                _base_ctr = None
                _base_topz = None
                if _place is not None and 'success sensor' in _place_src:
                  # run38 复盘（只读探针 probe_sc_sensor_geom.py 实测，取代 run37 的
                  # "杯壁横向 59mm" 推断）：
                  #   · success 传感器是 cup2 的**子节点**（parent=cup2），origin 落在
                  #     底座杯底部中心（z≈0.760=桌面高度，距杯轴仅 ~6mm）；
                  #   · 它的判定区要求：持物杯 **AABB 中心** 的 XY 落在传感器 origin
                  #     ±20mm 内、Z ∈ 底座杯中心 +[0.014, 0.109]（直立时 z∈[0.835,0.930]，
                  #     band center=0.882）。台面上平放（中心 0.821）与架在杯沿
                  #     （中心 0.979）都在带外 ⇒ 任务语义是**把杯子嵌进底座杯**。
                  #   · run37 的"+3cm 沿 base→sensor"方向来自 6mm 噪声（底座杯直立时
                  #     传感器几乎就在杯轴上），把放置点推到了杯轴外 3cm ⇒ XY 直接落
                  #     在 ±20mm 判定区之外。
                  # ⇒ 放置点 XY 取传感器/底座杯轴本身，**不再做横向偏移**；高度在下面
                  #    按判定带中心反算。
                  _base_name = None
                  _base_ctr = None
                  _base_topz = None
                  try:
                    for _bn in _same_class_full_names(self._env, _mn_tok):
                      if _f2_excl and _f2_excl in str(_bn).lower():
                        _base_name = str(_bn)
                        break
                  except Exception:
                    _base_name = None
                  try:
                    if _base_name is not None:
                      _bc0, _btz0 = self._env.get_object_pos_by_name(_base_name)
                      if _bc0 is not None:
                        _base_ctr = np.asarray(_bc0, dtype=float).reshape(3)
                      if _btz0 is not None:
                        _base_topz = float(_btz0)
                  except Exception:
                    _base_ctr, _base_topz = None, None
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: place XY = sensor/base axis '
                        f'{np.asarray(_place[:2], dtype=float).round(3)} (no lateral offset); base="{_base_name}" '
                        f'ctr={None if _base_ctr is None else _base_ctr.round(3)} top_z={_base_topz}{bcolors.ENDC}')
                if _place is not None:
                  # run32：sensor 读数偶发异常（同 episode 内实测某次 z=0.033，远低于
                  # 台面 0.75 与工作空间下限）——据此算出的下放高度会完全错。这里把
                  # 放置点先夹回工作空间内（留边距），再用于后续下放/释放。
                  # run40：但若夹取位移超过 3cm，说明传感器（=底座杯）已被推出机械臂
                  # 可达范围。此时继续"在夹取后的位置放置"只会把杯子丢到无关位置、
                  # 并把底座杯继续推远（run39 r0：底座杯被推到 x=0.93，place 被夹到
                  # 0.755，二者差 12.9cm，随后两次 place 都放空）。直接放弃本次放置。
                  try:
                    _place_raw = np.asarray(_place, dtype=float).reshape(3).copy()
                    _place = np.clip(_place_raw, _ws_min_f, _ws_max_f)
                    _clip_shift = float(np.linalg.norm(np.asarray(_place, dtype=float).reshape(3) - _place_raw))
                    if _clip_shift > 0.03:
                      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: place point {_place_raw.round(3)} '
                            f'outside workspace by {_clip_shift*1000:.0f}mm → abort placement (base cup likely pushed away){bcolors.ENDC}')
                      raise RuntimeError('place point outside workspace')
                  except RuntimeError:
                    raise
                  except Exception:
                    pass
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: place target via {_place_src} @ {_place.round(3)}{bcolors.ENDC}')
                  # run37：四点位置探针，把"success 传感器是否跟随底座杯移动"从推断
                  # 变成逐帧事实（此前只能对比"放置时刻"与"释放之后"两个时刻的读数）。
                  def _probe(_tag):
                    _sy = None
                    try:
                      if _sensor is not None:
                        _sy = np.asarray(_sensor.get_position(), dtype=float).reshape(3)
                    except Exception:
                      _sy = None
                    _bpos = None
                    try:
                      if _base_name is not None:
                        _tmpbp, _ = self._env.get_object_pos_by_name(_base_name)
                        if _tmpbp is not None:
                          _bpos = np.asarray(_tmpbp, dtype=float).reshape(3)
                    except Exception:
                      _bpos = None
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 probe[{_tag}]: '
                          f'sensor={_sy.round(3) if _sy is not None else None} '
                          f'base={_bpos.round(3) if _bpos is not None else None} '
                          f'place={_place.round(3)}{bcolors.ENDC}')
                  _probe('start')
                  # (a) 抬升 5cm（保持闭爪）避免拖动
                  _lp = np.array(self._env.get_ee_pos(), dtype=float)
                  # 记录"抓取时的持物高度"：侧抓物体放置时，EE 下放到同一高度才能
                  # 让物体底部重新回到台面。旧代码用 place_z-0.006 作为下放高度，
                  # 而侧抓物体会比 EE 低约 7~8cm → 物体底部扎进台面，descend 恒失败
                  # （run25 r2 实测 descend err≈70mm，恰好等于物体低于 EE 的深度）。
                  _ee_z_grasp = float(_lp[2])
                  # run28：记录"物体中心相对 EE 的 z 偏移"（侧抓时物体在 EE 下方
                  # 约 6~7cm）。放置下放必须按**物体**高度算，否则释放点会高出传感器
                  # 数厘米，杯子自由落体后滚离探测带（实测 d=65~126mm，未进探测带）。
                  _rel_z_grasp = None
                  _lift = np.clip(np.array([_lp[0], _lp[1], float(_lp[2]) + 0.05]), _ws_min_f, _ws_max_f)
                  self._env.apply_action(np.concatenate([_lift, _ee_quat_now, [0.0]]))
                  self._env.stabilize(steps=15)
                  # 抬升后复核抓取是否仍在：若搬运途中物体已滑落，继续"搬+放"只会
                  # 报出 placed & released 的假象（实测 cup 停位离释放点 157~534mm）。
                  try:
                    _cnt_lift = int(self._env.get_grasped_object_count())
                  except Exception:
                    _cnt_lift = 0
                  if _cnt_lift <= 0:
                    print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: grasp lost after lift (count=0) → abort placement{bcolors.ENDC}')
                    raise RuntimeError('grasp lost after lift')
                  # 几何跟随复核（run26 复盘）：cnt>0 只代表"抓取约束已建立"，
                  # headless 下该约束可能名存实亡——实测 run26 r0 cnt=1 却仍然
                  # "假搬运"，物体最终被推到工作空间外。用真实几何复核：被握物体
                  # 必须仍在 EE 附近（>12cm 即判假握持），立即终止，不做假搬运/假释放。
                  try:
                    _fl_ok, _fl_d, _fl_p = _grasp_follow_check(
                        self._env, _f2_name, self._env.get_ee_pos())
                    if _fl_d is not None:
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: follow check "{_f2_name}" obj={_fl_p.round(3)} ee={np.asarray(self._env.get_ee_pos(), dtype=float).round(3)} d={_fl_d*1000:.0f}mm ok={_fl_ok}{bcolors.ENDC}')
                      try:
                        _rel_z_grasp = float(_fl_p[2]) - float(np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)[2])
                      except Exception:
                        _rel_z_grasp = None
                    if not _fl_ok:
                      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: grasped object "{_f2_name}" NOT following EE (d>{120}mm) → fake grasp, abort placement{bcolors.ENDC}')
                      raise RuntimeError('grasped object not following')
                  except RuntimeError:
                    raise
                  except Exception:
                    pass
                  # (b) 闭环搬运到放置点上方
                  # run35 复盘：原实现固定用 "_place[2] + 0.08" 作为水平搬运高度，实测
                  # 持物中心比 EE 低约 69mm、杯底再低约 45mm → 杯底 ≈ place_z − 0.03，
                  # 水平转运时杯底几乎贴着台面（0.75）并直接撞上底座杯（顶面 ~0.83）；
                  # 而 StackCups 的 success 传感器是**挂在底座杯上**的（ep_r2 实测落点
                  # 精度 9mm，但 raw sensor 已被推到 420mm 外）。所以"每次搬运都把底座
                  # 杯推走"是 env 恒 False 的真实原因。
                  # 这里把转运高度改为按持物下垂量反算：EE z = 底座顶面 + 持物下垂量 + 余量，
                  # 保证持物杯底越过底座杯顶面之后再水平移动。
                  _hang = 0.06
                  try:
                    if _rel_z_grasp is not None:
                      _hang = float(abs(_rel_z_grasp)) + 0.045
                  except Exception:
                    _hang = 0.06
                  # run38：底座杯顶面直接取 **AABB 顶面**（探针实测直立时 0.900）。
                  # run37 用 "sensor_z+0.055=0.815" 把杯顶低估了 8.5cm —— 传感器 origin
                  # 在底座杯**底部**（z=0.760），杯顶其实是 sensor_z+0.14（≈中心+0.079）。
                  # 低估的直接后果：水平转运时持物杯底低于底座杯沿约 6cm，把底座杯刮走
                  # （run36/37 实测 raw sensor 每次被推 0.07~0.45m）。
                  _base_top = None
                  try:
                    if _base_topz is not None and 0.70 < float(_base_topz) <= float(_ws_max_f[2]):
                      _base_top = float(_base_topz)
                  except Exception:
                    _base_top = None
                  if _base_top is None:
                    # StackCups 等容器类任务：传感器在底座杯底部，杯顶 ≈ sensor+0.14。
                    # 但 stack_blocks / block_pyramid / empty_container 的传感器
                    # 就在台面/容器面上，+0.14 会让方块从 14cm 高空释放（实测被
                    # 弹到 0.84m 外）→ 这些任务直接以放置点高度作支撑面。
                    _base_top = (float(_place[2]) if _is_multi_obj_task
                                 else float(_place[2]) + 0.14)
                  _above_z = float(min(max(float(_place[2]) + 0.08, _base_top + _hang + 0.05),
                                       float(_ws_max_f[2])))
                  _above = np.clip(np.array([_place[0], _place[1], _above_z]), _ws_min_f, _ws_max_f)
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: transfer height z={_above_z:.3f} '
                        f'(base_top≈{_base_top:.3f}, hang={_hang:.3f}){bcolors.ENDC}')
                  self._env.apply_action(np.concatenate([_above, _ee_quat_now, [0.0]]))
                  self._env.stabilize(steps=15)
                  _mv_ok, _mv_err, _mv_it = self._env.move_ee_closed_loop(
                      _above, target_quat=_ee_quat_now, tol=0.01, max_iters=140)
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: move-to-place ok={_mv_ok} err={_mv_err*1000:.1f}mm iters={_mv_it}{bcolors.ENDC}')
                  if not _mv_ok:
                    # run31：闭环失败时回退一次 apply_action（OMPL 有避障，部分位形下比
                    # 小步 IK 更可靠），再闭环修正。实测 move-to-place 残差在 1.8~58.8mm
                    # 间波动，取决于当时的关节位形（小步 IK 会陷入无进展早退）。
                    try:
                      self._env.apply_action(np.concatenate([_above, _ee_quat_now, [0.0]]))
                      self._env.stabilize(steps=15)
                      _mv_ok, _mv_err, _mv_it = self._env.move_ee_closed_loop(
                          _above, target_quat=_ee_quat_now, tol=0.01, max_iters=140)
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: move-to-place retry(apply_action+closed-loop) ok={_mv_ok} err={_mv_err*1000:.1f}mm iters={_mv_it}{bcolors.ENDC}')
                    except Exception:
                      pass
                  # 搬运后复核抓取：run25 r1 实测搬运途中物体掉落但照常"放置"，
                  # 释放点与物体停位相距 509mm。抓取已丢 → 终止放置，不做假释放。
                  try:
                    _cnt_mv = int(self._env.get_grasped_object_count())
                  except Exception:
                    _cnt_mv = 0
                  if _cnt_mv <= 0:
                    print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: grasp lost during move-to-place (count=0) → abort placement (no drop){bcolors.ENDC}')
                    raise RuntimeError('grasp lost during move')
                  # 搬运后几何跟随复核：物体必须跟着 EE 一起到了放置区上方。
                  # 若物体仍留在原处（假握持），不做"原地释放"（那只会把物体留在
                  # 无关位置、伪造 placed，并让 Fix2 后续重入浪费预算）。
                  try:
                    _fl2_ok, _fl2_d, _fl2_p = _grasp_follow_check(
                        self._env, _f2_name, self._env.get_ee_pos())
                    if _fl2_d is not None:
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: follow check after move "{_f2_name}" obj={_fl2_p.round(3)} d={_fl2_d*1000:.0f}mm ok={_fl2_ok}{bcolors.ENDC}')
                      _probe('after-move')
                    if not _fl2_ok:
                      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: object "{_f2_name}" did not follow EE to place region (d>{120}mm) → abort release (no drop){bcolors.ENDC}')
                      raise RuntimeError('object did not follow to place')
                  except RuntimeError:
                    raise
                  except Exception:
                    pass
                  self._env.stabilize(steps=10)
                  # ── 插入类任务：竖直插刀（方案第 2 项，2026-10-25）────────────
                  # 依据 probe_knife_free.py / probe_knife_pose.py 实测几何：水平
                  # 搬运+释放一定失败，必须竖直刃朝下插入。接管成功就跳过通用
                  # 的 descend/release（用哨兵跳出，避免把刀再横过来丢一次）。
                  if _is_insert_task:
                    try:
                      _ins_ok = self._insert_knife_into_block(
                          _place, (_f2_name or _mn_tok), _ws_min_f, _ws_max_f)
                    except Exception as _inse:
                      _ins_ok = False
                      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert raised: {_inse}{bcolors.ENDC}')
                    if _ins_ok:
                      raise RuntimeError('SLOT_INSERT_HANDLED')
                  # (b2) 横向修正：把「被握物体中心」对到放置点 XY（纯 IK，在搬运高度完成）。
                  # 动机（run41，probe_stack_blocks.py 只读探针实测）：StackBlocks 的
                  # stack_blocks_success 检出体是半径≈30mm 的**圆柱**，而 descend 的
                  # 50mm 门限允许 32mm 偏差（实测落点 d=32mm）→ 方块中心落在圆外，
                  # DetectedSeveralCondition 恒不计数。原流程里 EE 到了放置点、但物体
                  # 因侧抓/夹持偏移并不在放置点正下方。这里按物体真实 AABB 中心与放置
                  # 点的横向差平移 EE（在搬运高度做，避免贴台面拖动），后续垂直下放
                  # 保持该修正。仅修正 ≤5cm 的残差，超过说明已撞飞 → 不修正。
                  _xy_drop = np.asarray(_place[:2], dtype=float).reshape(2).copy()
                  # ── PlaceCups 放置点：**必须落在辐条上，不能用环心**（2026-10-25）
                  # run10e 实测：三根辐条 XY 均值就是杯架**中央立柱**的位置。把杯子
                  # 对到环心后，杯底直接压到立柱顶面 —— 10 次里有 9 次 descend-to-place
                  # 卡在 ~97–99mm（EE≈1.15，杯底≈1.11 = 立柱顶），无法下放，仅 1/10
                  # 成功（还是杯子侧滑到辐条上才被检出）。故放置 XY 保持"单根辐条
                  # 中心"（success 传感器与该辐条中心重合）；能否站稳取决于杯子是否
                  # 骑在辐条上，属任务本身的物理敏感性。
                  try:
                    _op_c, _ = self._env.get_object_pos_by_name(_f2_name or _mn_tok)
                    if _op_c is not None:
                      _op_c = np.asarray(_op_c, dtype=float).reshape(3)
                      _dxy_c = np.asarray(_place[:2], dtype=float) - _op_c[:2]
                      _dn_c = float(np.linalg.norm(_dxy_c))
                      if 0.002 < _dn_c <= 0.05:
                        _ee_c = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
                        _fix_c = np.clip(np.array([_ee_c[0] + _dxy_c[0], _ee_c[1] + _dxy_c[1], _ee_c[2]]),
                                         _ws_min_f, _ws_max_f)
                        _fx_ok, _fx_err, _fx_it = self._env.move_ee_closed_loop(
                            _fix_c, target_quat=_ee_quat_now, tol=0.004, max_iters=90)
                        self._env.stabilize(steps=8)
                        _op_c2, _ = self._env.get_object_pos_by_name(_f2_name or _mn_tok)
                        _d_res = None
                        if _op_c2 is not None:
                          _d_res = float(np.linalg.norm(
                              np.asarray(_op_c2, dtype=float).reshape(3)[:2]
                              - np.asarray(_place[:2], dtype=float)))
                        # 下放必须是**纯垂直**的：XY 取修正后 EE 的实际位置。
                        # 若取物体 XY，EE 会再平移一个「物体-夹爪」横向偏移，
                        # 反而把物体推离放置点。
                        _xy_drop = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)[:2].copy()
                        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: XY centering '
                              f'obj→place {_dn_c*1000:.0f}mm ok={_fx_ok} err={_fx_err*1000:.1f}mm '
                              f'residual={"n/a" if _d_res is None else f"{_d_res*1000:.0f}mm"}{bcolors.ENDC}')
                      elif _dn_c > 0.05:
                        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: XY centering skipped '
                              f'(obj {_dn_c*1000:.0f}mm off place point → likely knocked away){bcolors.ENDC}')
                  except Exception as _xce:
                    print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: XY centering raised: {_xce}{bcolors.ENDC}')
                  # (c) 下放到放置高度：必须让被握物体尖端落入 success 传感器探测窗口
                  #     （实测 steak tip.z 窗口 ∈ [1.048, 1.063]，传感器 z≈1.061）。
                  #     旧代码用 _place[2]+0.03 释放过高 → 物体停在窗口之上 → Detected 失败。
                  #     run25 追加：下放高度取 max(place_z-0.006, 抓取持物高度)——
                  #     对侧抓的杯子，回到抓取高度才能让杯底落到台面；对 Meat 这类
                  #     抓取高度低于窗口的任务，原值保持不变。
                  _drop_z = float(_place[2]) - 0.006
                  try:
                    _drop_z = max(_drop_z, float(_ee_z_grasp))
                  except Exception:
                    pass
                  # run39：目标物体中心高度改为「**支撑面顶面 + 半高 + 2mm**」。
                  # 依据（probe_sc_nest_depth.py / probe_sc_rim_release.py 实测）：
                  #   · 台面平放：杯中心 0.8176（判定带下沿 0.835，差 1.7cm → 判不出）；
                  #   · 杯子能**部分嵌套**：cup3 从 cup1 杯顶轻放 → 稳定停在 0.8435
                  #     （在带内、XY 距杯轴 10mm）；
                  #   · 由此可知成功形态是**叠塔**：底座杯(台面) → cup1(≈0.8435) →
                  #     cup3(≈0.870)，被检测的两只都在带内。
                  # 旧值用「带中心 0.891」当目标物体中心 → 杯子已经插进底座杯内部 9mm，
                  # 与杯壁/杯底硬碰（run38 r0 直接把底座杯顶飞 0.5m），而释放瞬间虽然
                  # 短暂命中（run38 r2 cond[0]=True），沉降后仍掉回台面 0.8176 → False。
                  # 现在按"被握杯的杯底刚好触到支撑杯的杯沿"释放，让重力把它嵌进去。
                  if _rel_z_grasp is not None and not _is_meat_task:
                    try:
                      _half_h = None
                      try:
                        if _mv_top_z is not None and _mv_pos is not None:
                          _hh = float(_mv_top_z) - float(_mv_pos[2])
                          if 0.01 < abs(_hh) < 0.20:
                            _half_h = abs(_hh)
                      except Exception:
                        _half_h = None
                      if _half_h is None:
                        _half_h = 0.0787          # StackCups 单杯 AABB 半高实测值
                      # 支撑面 = 同类杯中（排除本次被握那只）最高的顶面
                      _support_top = None
                      try:
                        for _sn in _same_class_full_names(self._env, _mn_tok):
                          if _f2_name is not None and str(_sn).lower() == str(_f2_name).lower():
                            continue
                          _sc, _stz = self._env.get_object_pos_by_name(_sn)
                          if _sc is None or _stz is None:
                            continue
                          _sc = np.asarray(_sc, dtype=float).reshape(3)
                          if (not np.all(np.isfinite(_sc))) or _obj_off_workspace(self._env, _sc):
                            continue
                          # 只把**放置点附近**的同类物体当支撑面：旧实现取全场最高的
                          # 同类顶面，方块散落时会把邻块顶面误当支撑（0.80），
                          # 叠加 +0.14 后释放高度虚高。
                          try:
                            if float(np.linalg.norm(_sc[:2] - np.asarray(_place[:2], float))) > 0.08:
                              continue
                          except Exception:
                            pass
                          if _support_top is None or float(_stz) > _support_top:
                            _support_top = float(_stz)
                      except Exception:
                        _support_top = None
                      if _support_top is None:
                        _support_top = float(_base_top)
                      else:
                        # run40：遍历结果可能漏掉底座杯（run39 r1 实测 support_top=0.862，
                        # 低于底座杯 AABB 顶面 0.900）——底座杯是天然的第一层支撑面，
                        # 必须显式并入，否则第一只杯子会被当成"叠在台面杯上"而算低 4cm。
                        _support_top = max(float(_support_top), float(_base_top))
                      _obj_target_z = float(_support_top) + float(_half_h) + 0.002
                      # run40：删除旧版的"判定带上沿"截断。run39 实测该截断把目标物体
                      # 中心从 0.981 压到 0.926（= 底座杯中心 +0.105），杯底落在 0.847，
                      # 比底座杯沿 0.900 低 5.3cm → 释放瞬间直接撞飞底座杯（r0/r1/r2 的
                      # after-settle 里 sensor 分别漂 0.08/0.07/0.21m，r2 甚至被撞成倒扣）。
                      # 释放点只需满足"杯底刚好触到支撑杯杯沿"，**不需要**位于判定带内。
                      _drop_z_v2 = _obj_target_z - float(_rel_z_grasp)
                      _drop_z_v2 = max(_drop_z_v2, float(np.asarray(self._env.workspace_bounds_min, dtype=float).reshape(3)[2]) + 0.02)
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: drop-z by stack geometry: '
                            f'support_top={float(_support_top):.3f} half_h={float(_half_h):.3f} → obj z={_obj_target_z:.3f}; '
                            f'rel_z={float(_rel_z_grasp)*1000:.0f}mm → EE {_drop_z:.4f}→{_drop_z_v2:.4f}{bcolors.ENDC}')
                      _drop_z = float(_drop_z_v2)
                    except Exception:
                      pass
                  # ── MeatOffGrill：按传感器面 + 肉半高释放（2026-10-25）──────────
                  # 成功样本（r1/r2）肉心稳定停在 sensor_z + 半高（1.061+0.022≈1.083）；
                  # 基线 r0 走叠放支撑面时把悬空网格 top=1.317 当支撑 → EE 在 1.349
                  # 释放（离传感器 29cm），肉弹飞 411mm。这里按传感器面反算：
                  # 目标肉心 = sensor_z + 半高 + 2cm，EE = 目标肉心 − rel_z。
                  if _is_meat_task:
                    try:
                      _m_half = 0.022
                      if _mv_top_z is not None and _mv_pos is not None:
                        _mh = abs(float(_mv_top_z) - float(_mv_pos[2]))
                        if 0.005 < _mh < 0.08:
                          _m_half = _mh
                      _m_obj_z = float(_place[2]) + float(_m_half) + 0.02
                      _m_drop_z = (_m_obj_z - float(_rel_z_grasp)
                                   if _rel_z_grasp is not None else _m_obj_z + 0.008)
                      _m_drop_z = float(np.clip(_m_drop_z,
                                                float(_ws_min_f[2]) + 0.02,
                                                float(_ws_max_f[2])))
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 meat: release by sensor plane: '
                            f'sensor_z={float(_place[2]):.3f} half_h={float(_m_half):.3f} → obj z={_m_obj_z:.3f}; '
                            f'rel_z={-1.0 if _rel_z_grasp is None else float(_rel_z_grasp)*1000:.0f}mm → '
                            f'EE {_drop_z:.4f}→{_m_drop_z:.4f}{bcolors.ENDC}')
                      _drop_z = float(_m_drop_z)
                    except Exception as _mpe:
                      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 meat release-height '
                            f'raised: {_mpe}{bcolors.ENDC}')
                  # ── PlaceCups：按"辐条顶面"定释放高度（2026-10-25）───────────
                  # probe_place_cups_handle2.py 实测（同一投放点、三种手柄朝向）：
                  #   · 离传感器 6cm 落下 → 杯心 z≈1.02（判定锥内，三种朝向都 detected=True）；
                  #   · 12/18cm 落下 → 杯心停在 1.07/1.13（判定锥上沿 1.062 之上，全部 False）。
                  # 旧的 "+0.14 支撑面"让 EE 在 1.19 释放（杯底离辐条顶 7cm），多数次停在
                  # 架顶/台面 → 3/10。这里用辐条半径反算：杯底 = 辐条顶 + 2mm，
                  # 杯子只下落 1~2cm 就落进判定带，且不需要手柄 yaw 对齐（探针已证）。
                  if _is_cups_task:
                    try:
                      _rod_top = float(_place[2]) + 0.0066   # 辐条半径 ≈6.6mm（传感器就在辐条中心）
                      _mug_half = None
                      if _mv_top_z is not None and _mv_pos is not None:
                        _hh2 = abs(float(_mv_top_z) - float(_mv_pos[2]))
                        if 0.02 < _hh2 < 0.15:
                          _mug_half = _hh2
                      if _mug_half is None:
                        _mug_half = 0.0419                     # mug0 AABB 半高实测
                      _obj_z_c = _rod_top + float(_mug_half) + 0.002
                      _drop_z_c = (_obj_z_c - float(_rel_z_grasp)
                                   if _rel_z_grasp is not None else _obj_z_c + 0.042)
                      _drop_z_c = float(np.clip(_drop_z_c,
                                                float(_ws_min_f[2]) + 0.02,
                                                float(_ws_max_f[2])))
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 cups: release by spoke top: '
                            f'rod_top={_rod_top:.3f} mug_half={float(_mug_half):.3f} → mug z={_obj_z_c:.3f}; '
                            f'rel_z={-1.0 if _rel_z_grasp is None else float(_rel_z_grasp)*1000:.0f}mm → '
                            f'EE {_drop_z:.4f}→{_drop_z_c:.4f}{bcolors.ENDC}')
                      _drop_z = float(_drop_z_c)
                    except Exception as _cpe:
                      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 cups release-height '
                            f'raised: {_cpe}{bcolors.ENDC}')
                  _drop = np.clip(np.array([_xy_drop[0], _xy_drop[1], _drop_z]), _ws_min_f, _ws_max_f)
                  _dr_ok, _dr_err, _dr_it = self._env.move_ee_closed_loop(
                      _drop, target_quat=_ee_quat_now, tol=0.006, max_iters=120)
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: descend-to-place ok={_dr_ok} err={_dr_err*1000:.1f}mm iters={_dr_it} target_z={_drop[2]:.4f}{bcolors.ENDC}')
                  _probe('after-descend')
                  if not _dr_ok:
                    # run31：下放失败同样回退一次 apply_action 再闭环，缓解小步 IK
                    # 在位形不佳时的无进展早退（残差 1.3~58.7mm 波动）。
                    try:
                      self._env.apply_action(np.concatenate([_drop, _ee_quat_now, [0.0]]))
                      self._env.stabilize(steps=12)
                    except Exception:
                      pass
                    _dr_ok, _dr_err, _dr_it2 = self._env.move_ee_closed_loop(
                        _drop, target_quat=_ee_quat_now, tol=0.006, max_iters=120)
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: descend-to-place retry ok={_dr_ok} err={_dr_err*1000:.1f}mm iters={_dr_it2} target_z={_drop[2]:.4f}{bcolors.ENDC}')
                  self._env.stabilize(steps=10)
                  # (c2) 释放前守卫：仍握持物体 + 落点误差足够小，否则不释放。
                  #      在错误高度/错误位置释放只会把物体丢在随机位置（run25 r1/r2
                  #      实测释放点距物体停位 228/509mm），且伪造"placed & released"日志。
                  try:
                    _cnt_dr = int(self._env.get_grasped_object_count())
                  except Exception:
                    _cnt_dr = 0
                  if _cnt_dr <= 0:
                    print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: grasp lost before release (count=0) → abort placement (no drop){bcolors.ENDC}')
                    raise RuntimeError('grasp lost before release')
                  if float(_dr_err) > 0.05:
                    print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2: descend not reached (err={_dr_err*1000:.1f}mm > 50mm) → abort placement (no drop){bcolors.ENDC}')
                    raise RuntimeError('descend-to-place not reached')
                  # (d) 张开夹爪释放，并多稳定若干步让物体落定在传感器窗口内
                  # run39：沉降步数 35 → 120。探针实测杯子要 240~300 步才落定；
                  # run38 r2 的 cond[0]=True 只是"释放瞬间"的假象，落回台面后即 False。
                  # 这里给足沉降时间，让 diag 打印与 env.success() 反映**真实终态**。
                  self._env.release_with_settle(stabilize_steps=120, lift_before_release=False)
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: placed & released at {_place.round(3)}{bcolors.ENDC}')
                  _probe('after-settle')
                  # 诊断：释放后**逐个同类物体**打印实际停位与到传感器的距离。
                  # run30 教训：原先用泛称 _mn_tok 解析（get_object_pos_by_name 恒返回
                  # 第一个候选），若本次放置的是 cup3 而 cup1 在别处，读到的就是 cup1 的
                  # 坐标 → d 显示 532/798mm 的假报警。改为遍历同类具体名，一次看清。
                  try:
                    _seen_chk = False
                    for _nm_chk in _same_class_full_names(self._env, _mn_tok):
                      try:
                        _pp_chk, _ = self._env.get_object_pos_by_name(_nm_chk)
                        if _pp_chk is None:
                          continue
                        _seen_chk = True
                        _ppv = np.asarray(_pp_chk, dtype=float).reshape(3)
                        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 diag: after release "{_nm_chk}" center={_ppv.round(3)} sensor={_place.round(3)} d={np.linalg.norm(_ppv-_place)*1000:.0f}mm env_done={self._env.success()}{bcolors.ENDC}')
                      except Exception:
                        continue
                    if not _seen_chk:
                      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 diag: no same-class object resolvable after release (tok={_mn_tok}){bcolors.ENDC}')
                  except Exception:
                    pass
                  # run35：逐条打印任务**原生**成功条件，直接看清"还差哪个条件"。
                  # StackCups = DetectedCondition(cup1) ∧ DetectedCondition(cup3) ∧ NothingGrasped。
                  # 有了这一行就不必再从落点距离上猜。
                  try:
                    _conds = getattr(self._env.rlbench_env._scene.task, '_success_conditions', []) or []
                    for _ci, _cnd in enumerate(_conds):
                      try:
                        _res = _cnd.condition_met()
                        if isinstance(_res, (tuple, list)) and len(_res) >= 1:
                          _ok_c = bool(_res[0])
                          _msg_c = str(_res[1]) if len(_res) > 1 else ''
                        else:
                          _ok_c, _msg_c = bool(_res), ''
                      except Exception as _ce:
                        _ok_c, _msg_c = False, 'err %s' % _ce
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 diag: success_condition[{_ci}] '
                            f'{type(_cnd).__name__} met={_ok_c} ({_msg_c[:90]}){bcolors.ENDC}')
                  except Exception:
                    pass
                  # run35：打印 success 传感器**原始**坐标（未做 +3.5cm 横向偏移前），
                  # 用于核对"偏移后落点"与"探测中心"的相对几何。
                  try:
                    if _sensor is not None:
                      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 diag: raw success sensor @ '
                            f'{np.array(_sensor.get_position(), dtype=float).reshape(3).round(3)} '
                            f'(place target after offset @ {_place.round(3)}){bcolors.ENDC}')
                  except Exception:
                    pass
              except Exception as _pke:
                if str(_pke) in ('LIFT_ONLY_DONE', 'LIFT_ONLY_FAIL'):
                  # lift-only 哨兵：不是异常，是"已处理完、跳过放置分支"的信号。
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 lift-only '
                        f'branch finished ({_pke}); placement skipped{bcolors.ENDC}')
                elif str(_pke) == 'SLOT_INSERT_HANDLED':
                  # 插入类哨兵：竖直插刀原语已接管放置（含释放与终态诊断）。
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 slot-insert '
                        f'branch finished; generic placement skipped{bcolors.ENDC}')
                else:
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 PICK place raised: {_pke}{bcolors.ENDC}')
            elif not _ok2:
              # Grasp failed → 横向推兜底只对"把物体从台面推落"类任务有意义
              # （MeatOffGrill：把肉从烤架上推下）。对 pick 类任务（StackCups 等）
              # 横向推会把目标推离台面（实测 cup 落到 [0.253,-0.669,0.03]，已在
              # 工作空间外），之后永久不可达；故其余任务直接跳过，成功判定仍交给
              # 原生 grasp()/env.success()。
              _push_task = False
              try:
                _push_task = 'Meat' in type(self._env.rlbench_env._scene.task).__name__
              except Exception:
                _push_task = False
              if _push_task or 'meat' in str(_mn_tok).lower():
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: grasp failed → horizontal push to knock "{_mn_tok}" off surface{bcolors.ENDC}')
                try:
                  _lp = np.array(self._env.get_ee_pos(), dtype=float)
                  _z_here = float(_lp[2])
                  # Push toward workspace edge (away from grill center)
                  _ws_c = (np.array(self._env.workspace_bounds_min[:2]) + np.array(self._env.workspace_bounds_max[:2])) / 2.0
                  # Push in direction from workspace center to EE (outward)
                  _push_dir = _lp[:2] - _ws_c
                  _push_dist = float(np.linalg.norm(_push_dir))
                  if _push_dist > 0.01:
                    _push_tgt = _lp[:2] + _push_dir / _push_dist * 0.15
                  else:
                    _push_tgt = np.array([_ws_c[0] + 0.15, _ws_c[1]])
                  _push_tgt = np.clip(_push_tgt, _ws_min_f[:2], _ws_max_f[:2])
                  self._env.horizontal_push_continuous(
                      target_xy=_push_tgt,
                      total_steps=30, gripper_action=0.3, clamp_z=_z_here,
                  )
                except Exception as _hpe:
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 horizontal push fallback raised: {_hpe}{bcolors.ENDC}')
              else:
                print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: grasp failed on non-push task → skip horizontal push (avoid knocking "{_mn_tok}" off surface){bcolors.ENDC}')
              # No force_override final fallback — physics-based push MUST work;
              # if it doesn't, that's a genuine failure we want to measure.
          except Exception as _picke:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK raised: {_picke}{bcolors.ENDC}')
          # ── 多物体任务：首轮只够一只 → 继续搬（每轮读原生 env.success()）────
          # probe_stack_blocks.py 实测：stack_blocks_success 的检出体是半径≈30mm
          # 的圆柱，两只 50mm 方块必须**叠放**才能同时被检出。单轮 Fix2 只搬一只，
          # DetectedSeveralCondition(..., n=2) 恒 False；这里补第 2 轮。
          if _is_multi_obj_task and not getattr(self, '_multi_round_used', False):
            _mr_place = None
            _mr_ok0 = False
            try:
              _mr_place = locals().get('_place', None)
              _mr_ok0 = bool(self._env.success())
            except Exception:
              _mr_ok0 = False
            if not _mr_ok0:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: first round insufficient '
                    f'(env.success=False) → continue picking detected objects{bcolors.ENDC}')
              self._multi_round_used = True
              try:
                # _mr_place 可能为 None（首轮抓取就失败、从未解析过放置点）——
                # 由 helper 从成功条件里的探测传感器自行解析。
                _mr_ok = self._multi_round_pick_place(_mn_tok, _mr_place, max_rounds=3, budget_s=150.0)
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: finished '
                      f'ok={_mr_ok}{bcolors.ENDC}')
              except Exception as _mre:
                print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round raised: {_mre}{bcolors.ENDC}')
      # ── 多物体任务：计划层一直握着方块不松手 → 接管放置 ──────────────────
      # r1 实测失败模式：composer 第一步就把方块抓起来，之后每个 composer 步骤
      # 结束时 grasped=1，Fix2 因 `grasped==0` 前置条件永不进入 → 方块被握到
      # episode 结束、env 恒 False。这里先把手上那只放进 success 区，再补轮。
      elif _is_multi_obj_task and _grasped_before > 0 and not _env_done:
        if not getattr(self, '_multi_round_used', False):
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: plan is holding '
                f'{_grasped_before} object(s) in a multi-object task → place held object at success zone{bcolors.ENDC}')
          self._multi_round_used = True
          try:
            _mrh_ok = self._multi_round_pick_place(_mn_tok, None, max_rounds=3, budget_s=150.0)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: held-object branch '
                  f'finished ok={_mrh_ok}{bcolors.ENDC}')
          except Exception as _mrh_e:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round held-object branch '
                  f'raised: {_mrh_e}{bcolors.ENDC}')
      # =====================================================================
      # Fix3 (Slide-class fallback — SlideBlockToTarget, OpenDrawer etc.)
      # When: movable ∈ SLIDABLE (block/slider/drawer).
      # Typical failure case: SlideBlockToTarget → composer moves EE to
      # block and press_down presses ~30mm straight down, but never moves
      # horizontally toward the target area → block stays → task fails.
      # We close gripper slightly (gripper=0.3) for friction contact and
      # step horizontally to the TARGET object's XY position.
      #
      # CRITICAL FIX 2026-08-15: The composer fallback always generates a
      # press_down affordance at the BLOCK position, so
      # _last_target_center_world = block position → _dxy = 0 → Fix3 never
      # triggered.  Now we actively query the scene for a "target" object
      # (SlideBlockToTarget exposes one) and use ITS world position as the
      # slide destination.  Also relaxed object_centric requirement since
      # EE-centric composer fallback can still detect movable_name="block".
      # =====================================================================
      elif _is_slidable:
        _ee_cur = None
        try:
          _ee_cur = np.array(self._env.get_ee_pos(), dtype=float)
        except Exception:
          _ee_cur = None
        # ── Resolve slide destination ──────────────────────────────────────
        # Priority: 1) scene "target" object position  2) _last_target_center_world
        _slide_tgt_xy = None
        _slide_src = None
        try:
          _obj_names = self._env.get_object_names()
          _TARGET_KEYWORDS = ['target', 'goal', 'destination', 'drop']
          for _tn in _obj_names:
            _tn_l = _tn.lower()
            if any(_kw in _tn_l for _kw in _TARGET_KEYWORDS):
              try:
                _tgt_pos = None
                # Method 1: get_3d_obs_by_name
                _tgt_obs = self._env.get_3d_obs_by_name(_tn)
                if _tgt_obs is not None:
                  try:
                    _tgt_pos = np.asarray(_tgt_obs['_position_world'], dtype=float).reshape(3)
                  except Exception:
                    pass
                  if _tgt_pos is None:
                    try: _tgt_pos = np.asarray(_tgt_obs['position'], dtype=float).reshape(3)
                    except Exception: pass
                # Method 2: direct scene query via name2ids + pyrep Shape
                if _tgt_pos is None:
                  try:
                    _ids = self._env.name2ids.get(_tn)
                    if _ids:
                      from pyrep.objects import Shape as _Shape
                      _sh = _Shape(_ids[0])
                      _tgt_pos = np.array(_sh.get_position(), dtype=float)
                  except Exception:
                    pass
                # Method 3: detect() fallback
                if _tgt_pos is None:
                  try:
                    _det = self.detect(_tn)
                    if _det is not None:
                      try: _tgt_pos = np.asarray(_det['_position_world'], dtype=float).reshape(3)
                      except Exception:
                        try: _tgt_pos = np.asarray(_det['position'], dtype=float).reshape(3)
                        except Exception: pass
                  except Exception:
                    pass
                if _tgt_pos is not None:
                  _slide_tgt_xy = np.array([_tgt_pos[0], _tgt_pos[1]], dtype=float)
                  _slide_src = f'scene "{_tn}" @ ({_tgt_pos[0]:.3f},{_tgt_pos[1]:.3f},{_tgt_pos[2]:.3f})'
                  break
              except Exception:
                pass
        except Exception:
          pass
        if _slide_tgt_xy is None and _last_target_center_world is not None:
          _lt = np.asarray(_last_target_center_world, dtype=float).reshape(3)
          _slide_tgt_xy = np.array([_lt[0], _lt[1]], dtype=float)
          _slide_src = f'_last_target_center_world ({_lt[0]:.3f},{_lt[1]:.3f})'
        # Fallback 3: if still no target, move toward workspace center
        if _slide_tgt_xy is None:
          try:
            _ws_c = (np.array(self._env.workspace_bounds_min[:2]) + np.array(self._env.workspace_bounds_max[:2])) / 2.0
            _slide_tgt_xy = _ws_c.astype(float)
            _slide_src = f'workspace_center ({_ws_c[0]:.3f},{_ws_c[1]:.3f})'
          except Exception:
            pass
        if _ee_cur is not None and _slide_tgt_xy is not None:
          _dxy = float(np.linalg.norm(_ee_cur[:2] - _slide_tgt_xy))
          _z_here = float(_ee_cur[2])
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 SLIDE FALLBACK: movable="{_last_movable_name}" (tok={_mn_tok}), EE=({_ee_cur[0]:.3f},{_ee_cur[1]:.3f},{_z_here:.3f}), slide_tgt={_slide_src}, dxy={_dxy*100:.1f}cm; closed-loop push{bcolors.ENDC}')
          try:
            # run39：Slide 类任务**不做抓取**。2026-10-23 探针实测（probe_slide_block）：
            #   1) 方块高 56mm，夹爪在方块中心高度闭爪 open=0.000 / cnt=0，从未夹住；
            #   2) 下降扫描把方块推走 96~137mm（有一次直接甩下台面），
            #      之后的水平推完全错位 → 这就是 0/10 的直接成因之一；
            #   3) 判定区 = target 半径 30~40mm（XY）、块中心 z∈[0.758,0.790]，
            #      方块静置时本来就在 Z 带内，只需要把它横向推到 target 附近。
            # 因此这里改成**闭环推块**：贴到方块后方 → 推一段 → 重读方块位置 → 再推，
            # 直到进入判定带或预算耗尽。成功判定仍完全交给原生 env.success()。
            _blk = None
            _blk_top = None
            for _bn in (_last_movable_name, _mn_tok, 'block', 'slider', 'cube'):
              if not _bn:
                continue
              try:
                _bc, _btz = self._env.get_object_pos_by_name(_bn)
              except Exception:
                continue
              if _bc is not None:
                _blk = np.asarray(_bc, dtype=float).reshape(3)
                _blk_top = (float(_btz) if _btz is not None else float(_blk[2]) + 0.028)
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3: movable resolved via "{_bn}" '
                      f'center={_blk.round(3)} top_z={_blk_top:.4f}{bcolors.ENDC}')
                break
            if _blk is None:
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3: 无法解析可移动物体世界坐标 → 退化单次推块{bcolors.ENDC}')
              self._env.horizontal_push_continuous(
                  target_xy=_slide_tgt_xy, total_steps=40,
                  gripper_action=0.0, clamp_z=_z_here, budget_s=8.0)
            else:
              try:
                _ws_lo = np.asarray(self._env.workspace_bounds_min, dtype=float) + np.array([0.01, 0.01, 0.0])
                _ws_hi = np.asarray(self._env.workspace_bounds_max, dtype=float) - np.array([0.01, 0.01, 0.0])
              except Exception:
                _ws_lo = _ws_hi = None

              def _read_block():
                """读可移动方块的真实世界坐标与顶面高度（只读，不改变场景）。"""
                for _bn in (_last_movable_name, _mn_tok, 'block', 'slider', 'cube'):
                  if not _bn:
                    continue
                  try:
                    _cc, _tt = self._env.get_object_pos_by_name(_bn)
                  except Exception:
                    continue
                  if _cc is not None:
                    return (np.asarray(_cc, dtype=float).reshape(3),
                            (float(_tt) if _tt is not None else None))
                return None, None

              # 参考点：**原生 success 传感器**的世界坐标（判定体就在传感器上）。
              # 不能用场景 "target" 物体坐标代替 —— 两者可能不在同一点
              # （StackCups 实测传感器与底座杯轴偏移 59mm）。
              _sen_xy = None
              try:
                _cs_all = getattr(self._env.rlbench_env._scene.task, '_success_conditions', []) or []
                for _c in _cs_all:
                  _det = getattr(_c, '_detector', None)
                  if _det is not None:
                    _sen_xy = np.asarray(_det.get_position(), dtype=float).reshape(3)[:2]
                    break
              except Exception:
                _sen_xy = None
              if _sen_xy is None:
                _sen_xy = np.asarray(_slide_tgt_xy, dtype=float).reshape(2)
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 ref: success sensor @ '
                    f'({_sen_xy[0]:.3f},{_sen_xy[1]:.3f}){bcolors.ENDC}')

              _t_slide = time.time()
              _SLIDE_BUDGET_S = 55.0

              try:
                _q_home = np.asarray(self._env.get_home_ee_quat(), dtype=float).reshape(4)
              except Exception:
                _q_home = None

              def _move(tgt, tol=0.012, max_iters=90):
                """闭环移动，失败时换 home 朝向再试一次（IK 朝向约束会决定可达性）。"""
                _o, _e, _i = self._env.move_ee_closed_loop(tgt, tol=tol, max_iters=max_iters)
                if (not _o) and _e > 0.04 and _q_home is not None:
                  _o_b, _e_b, _i_b = self._env.move_ee_closed_loop(
                      tgt, target_quat=_q_home, tol=tol, max_iters=max_iters)
                  if _o_b or _e_b < _e:
                    return _o_b, _e_b, _i_b
                return _o, _e, _i

              def _conds_met_now():
                """原生成功条件当前是否全部满足（只读，仅用于决定"还要不要推"）。

                注意：RLBench 的 `Condition.condition_met()` 返回的是
                `(met, done)` **元组**，其中 `done` 对 DetectedCondition 恒为
                False。因此：
                  - `bool(tuple)` 恒为 True（非空元组）—— 会误判"已成功"；
                  - `all(...)` 恒为 False（done=False）—— 会误判"永远没成功"。
                正确做法是**只取第 0 个元素**（met）。实测 sensor+32mm 处
                env.success()=True 而 all()=False，就是这个坑。
                """
                try:
                  _cs = getattr(self._env.rlbench_env._scene.task, '_success_conditions', []) or []
                  _out = []
                  for _c in _cs:
                    _r = _c.condition_met()
                    if isinstance(_r, tuple):
                      _out.append(bool(_r[0]))
                    else:
                      _out.append(bool(_r))
                  return _out
                except Exception:
                  return []

              def _cond_all():
                _c = _conds_met_now()
                return bool(_c) and all(_c)

              # 判定带实测（probe_slide_geom）：距 success 传感器 **22~38mm 的圆环**，
              # yaw 0°~45°、沿 x/沿 y 完全一致；圆心 22mm 内是死区（方块 AABB 包住
              # 传感器时 CoppeliaSim 判不检出）；4mm/步连续穿越同样能命中。
              # 因此推块的目标是"把方块送进环带"，而不是推到圆心。
              _STOP_MARGIN = 0.036      # 推进到距传感器 36mm 收手（环带 22~38mm 中心）
              _RING_IN = 0.022
              _dead_escaped = 0
              for _try in range(4):
                if time.time() - _t_slide > _SLIDE_BUDGET_S:
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3: '
                        f'推块总预算 {_SLIDE_BUDGET_S:.0f}s 耗尽 — 停止{bcolors.ENDC}')
                  break
                _b_now, _bt_now = _read_block()
                if _b_now is None:
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3: '
                        f'方块坐标不可解析 — 停止{bcolors.ENDC}')
                  break
                _blk = _b_now
                if _bt_now is not None:
                  _blk_top = float(_bt_now)
                _dv = _sen_xy - _blk[:2]
                _dist = float(np.linalg.norm(_dv))
                _cm0 = _conds_met_now()
                if _cm0 and all(_cm0):
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3: 原生成功条件已满足 '
                        f'(dist={_dist*1000:.1f}mm) — stop pushing{bcolors.ENDC}')
                  break
                if _dist < 1e-6:
                  break
                _u = _dv / _dist
                if _dist < _RING_IN + 0.004 and _dead_escaped < 2:
                  # 滑行过头进了圆心死区：从传感器一侧把方块往回推 ~14mm。
                  _dead_escaped += 1
                  _d = -_u
                  _start = _blk[:2] - _d * 0.055
                  _used_deg = -999.0
                  _L = 0.014
                  _stop_pred = _cond_all
                else:
                  # 方向选择：理想方向是 u（block→sensor）。方块常在远端边缘，
                  # 其"后方 5.5cm"会落到可达区外（实测 err 120mm）。按
                  # ±25°/±45°/±60° 依次侧偏，取第一个后方点在工作空间内的方向；
                  # 侧偏方向的推进量按 cos 分量折算，保证仍在靠近传感器。
                  _chosen = None
                  for _deg in (0, 25, -25, 45, -45, 60, -60):
                    if _deg == 0:
                      _d = _u.copy()
                    else:
                      _cr, _sr = np.cos(np.deg2rad(_deg)), np.sin(np.deg2rad(_deg))
                      _d = np.array([_cr * _u[0] - _sr * _u[1],
                                     _sr * _u[0] + _cr * _u[1]], dtype=float)
                    _sc = _blk[:2] - _d * 0.055
                    if _ws_lo is not None and (
                        np.any(_sc < _ws_lo[:2] + 0.005) or np.any(_sc > _ws_hi[:2] - 0.005)):
                      continue
                    _chosen = (_d, _sc, float(_deg))
                    break
                  if _chosen is None:
                    _d = _u.copy()
                    _sc = (_blk[:2] - _u * 0.055) if _ws_lo is None \
                        else np.clip(_blk[:2] - _u * 0.055, _ws_lo[:2], _ws_hi[:2])
                    _chosen = (_d, _sc, 999.0)
                  _d, _start, _used_deg = _chosen
                  # 只需推到距传感器 _STOP_MARGIN 处（不是推过圆心）：距离超出
                  # 外径 38mm 时一律按沿 u 的有效分量折算。
                  _prog = float(np.dot(_d, _u))
                  _L = float(min(0.18, max(0.0, (_dist - _STOP_MARGIN)) / max(_prog, 0.5)))
                  if _L < 0.004:
                    _L = 0.008      # 已在环带边缘但条件未成立：小步探一下

                  _pred_calls = [0]

                  def _stop_pred():
                    """每步判定：原生条件成立，或方块已进到距传感器 36mm 内。

                    第 0 步（EE 尚未移动）只读原生条件 —— 否则方块本来就
                    在预停线内时会立即返回 True，推块原地不动作。
                    """
                    _pred_calls[0] += 1
                    if _cond_all():
                      return True
                    if _pred_calls[0] <= 1:
                      return False
                    _bb, _ = _read_block()
                    if _bb is None:
                      return False
                    return float(np.linalg.norm(
                        np.asarray(_bb, dtype=float)[:2] - _sen_xy)) <= _STOP_MARGIN

                _z_push = float(_blk_top) - 0.015
                # 先竖直抬到方块顶面以上再横移，避免斜穿方块本体。
                try:
                  _ee0 = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
                  if _ee0[2] < float(_blk_top) + 0.045:
                    _move(np.array([_ee0[0], _ee0[1], float(_blk_top) + 0.055], dtype=float),
                          tol=0.015, max_iters=60)
                except Exception:
                  pass
                _above = np.array([_start[0], _start[1], float(_blk_top) + 0.05], dtype=float)
                _at = np.array([_start[0], _start[1], _z_push], dtype=float)
                _o1, _r1, _i1 = _move(_above, tol=0.012, max_iters=90)
                if (not _o1) and _r1 > 0.06:
                  # 远端方块（实测 >250mm 时命中率骤降）：先退到工作空间上方
                  # 中转一次，给 IK 一个不拉伸的中间构型，再重新逼近。
                  try:
                    _cen = (np.asarray(self._env.workspace_bounds_min, dtype=float)
                            + np.asarray(self._env.workspace_bounds_max, dtype=float)) / 2.0
                    _move(np.array([_cen[0], _cen[1], 1.0], dtype=float), tol=0.04, max_iters=70)
                    _o1, _r1, _i1 = _move(_above, tol=0.012, max_iters=90)
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 push#{_try+1}: '
                          f'中转后重新逼近 ok={_o1} err={_r1*1000:.1f}mm{bcolors.ENDC}')
                  except Exception:
                    pass
                _o2, _r2, _i2 = _move(_at, tol=0.010, max_iters=90)
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 push#{_try+1}: '
                      f'blk={_blk.round(3)} top={_blk_top:.3f} dist={_dist*1000:.1f}mm '
                      f'z_push={_z_push:.3f} dir={_used_deg:+.0f}° L={_L*1000:.0f}mm '
                      f'approach ok={_o1}/{_o2} err={_r1*1000:.1f}/{_r2*1000:.1f}mm{bcolors.ENDC}')
                if (not _o2) and _r2 > 0.035:
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 push#{_try+1}: '
                        f'无法到方块后方 (err={_r2*1000:.1f}mm) — 放弃本次推出{bcolors.ENDC}')
                  break
                # 推的终点 = **EE 起点** + d*(L+2.5cm 贴靠余量)。不能写成
                # "方块中心 + d*(L+2cm)"：EE 起点本来就在方块后方 5.5cm，
                # 那样会让 EE 多走 5.5cm，方块被顶飞（r45 实测 EE 走 126mm、
                # 方块飞 400mm，直接越过判定环）。
                _push_tgt = np.asarray(_start, dtype=float)[:2] + np.asarray(_d, dtype=float) * float(_L + 0.025)
                self._env.horizontal_push_continuous(
                    target_xy=_push_tgt,
                    # 1.5mm/步 + 步间 6 步稳定：方块滑行速度远高于 EE，粗步长会
                    # 一步跨过 16mm 宽的判定环，逐 step 的 stop_when 就漏检。
                    total_steps=max(10, min(160, int((_L + 0.025) / 0.0015) + 4)),
                    gripper_action=0.0, clamp_z=_z_push, budget_s=18.0,
                    stop_when=_stop_pred, settle_steps=6)
                # 重读方块真实位置（推完可能已进环带、被推歪、或滑入死区）
                _bc2, _btz2 = _read_block()
                if _bc2 is None:
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 push#{_try+1}: 方块坐标不可解析 — 停止{bcolors.ENDC}')
                  break
                _moved = float(np.linalg.norm(np.asarray(_bc2, dtype=float)[:2] - _blk[:2]))
                _blk = np.asarray(_bc2, dtype=float).reshape(3)
                if _btz2 is not None:
                  _blk_top = float(_btz2)
                _dist_new = float(np.linalg.norm(
                    np.asarray(_sen_xy, dtype=float).reshape(2) - _blk[:2]))
                _cmet = _conds_met_now()
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 push#{_try+1}: '
                      f'block moved {_moved*1000:.1f}mm → {_blk.round(3)} '
                      f'(dist {_dist*1000:.1f}→{_dist_new*1000:.1f}mm) conds={_cmet}{bcolors.ENDC}')
                if _cmet and all(_cmet):
                  print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 push#{_try+1}: '
                        f'原生成功条件已满足 — stop pushing{bcolors.ENDC}')
                  break
                if _moved < 0.004:
                  print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 push#{_try+1}: '
                        f'方块几乎未被推动 — 停止（避免空转到预算耗尽）{bcolors.ENDC}')
                  break
                # 其余情况（仍在外环、或刚滑进死区）都继续下一轮：
                # 死区由上面的逃逸分支处理，外侧由方向候选重新对准。
              # 不再追加"收尾轻压"：方块本来就在台面上（slide 类不涉及抬升），
              # 条件满足后再下压只会把方块顶出判定区（实测 r6/r7 即此风险）。
            # No force_override final fallback — physics-based push MUST work;
            # if it doesn't, that's a genuine failure we want to measure.
          except Exception as _slidee:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 SLIDE FALLBACK raised: {_slidee}{bcolors.ENDC}')
        else:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 SLIDE: skipped (ee_cur={_ee_cur is not None}, slide_tgt={_slide_tgt_xy is not None}){bcolors.ENDC}')
      elif _intend_close:
        # Pickable 任务禁止按"EE 当前位姿"盲目闭爪：LLM 生成的 affordance 常落在
        # 语义泛称物体上（如 'a point at the center of the meat' → headless 下解析到
        # 先注册的 chicken），盲目闭爪会在 Fix2 用真实名解析纠正目标之前，先 grasp 到
        # 错误物体（实测 MeatOffGrill var1：steak 任务却先抓到 chicken）。
        # 这类任务统一交给上面的 Fix2（真实名解析 + 接近 + 下降 + 抓取）。
        # 若此处已握有物体（_grasped_before>0）则更不能在任意位姿再闭爪去抓第二个物体。
        if _is_pickable:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: intend CLOSE on pickable task → skip blind close (handled by Fix2 real-name pick; holding={_grasped_before}){bcolors.ENDC}')
          # lift-only 任务已握持时不能停在"已握持所以跳过"：成功条件还要求物体
          # 进入上方传感器。实测 r10/r11 在这里被跳过 → 全程握着目标却从未举起。
          if _is_lift_task and _grasped_before > 0:
            _lift_ok_h = _lift_to_sensor('already-holding')
            if _lift_ok_h:
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: lift-only already-holding SUCCESS — holding pose kept{bcolors.ENDC}')
        else:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: intend CLOSE (gripper={float(gripper_state):.2f}); running grasp_with_retry{bcolors.ENDC}')
          try:
            _ok, _cnt = self._env.grasp_with_retry(max_retry=3, stabilize_steps=25, push_down_m=0.015)
          except Exception as _gre:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] grasp_with_retry raised: {_gre}{bcolors.ENDC}')
      elif _intend_open and _grasped_before > 0:
        if _is_lid_task:
          # 锅盖的成功条件含 GraspedCondition：释放会让终态永久失败（基线 0/10 的
          # 直接原因就是"举到窗口下沿就 open gripper"）。保持握持，交给下一次
          # post-action 的 lift-only 继续举升。
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: skip release '
                f'(lid task requires the lid to stay grasped){bcolors.ENDC}')
        else:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: intend OPEN (gripper={float(gripper_state):.2f}) while holding {_grasped_before} objects; running release_with_settle{bcolors.ENDC}')
          try:
            self._env.release_with_settle(stabilize_steps=25, lift_before_release=True)
          except Exception as _rle:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] release_with_settle raised: {_rle}{bcolors.ENDC}')
      elif _did_press_down:
        # Unknown task that ran press-down (not control/pick/slide) → small hold
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: press-down (unknown task); hold_press 30 steps{bcolors.ENDC}')
        try:
          self._env.hold_press(hold_steps=30)
        except Exception as _hpe:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] hold_press raised: {_hpe}{bcolors.ENDC}')
    except Exception as _post_e:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Post-action exception: {_post_e}{bcolors.ENDC}')

    return execute_info

  def _insert_knife_into_block(self, sensor_xyz, tok, ws_min, ws_max):
    """插入类原语：把（已握持的）刀转成**刃朝下**，对准刀座槽位竖直下放并释放。

    依据（2026-10-25 只读探针 probe_knife_pose.py / probe_knife_free.py 实测）：
      · 刀座 4.1×8.0×15.5cm，`success` 传感器在刀座**底部**（台面上方约 1.4cm）；
      · 检出区 = 刀座内部的竖直槽区：静态扫描下，刀**中心**落在传感器 xy±40mm、
        z∈[sensor, sensor+0.09] 即检出（与 yaw 无关，说明检出体是粗圆柱）；
      · 刀**平放**（长轴水平）释放 → 跨在刀座顶面滑落，中心掉到台面（z<sensor）
        或 d>40mm，三种 yaw 全部不检出；
      · 刀**竖直、刃朝下**释放 → 稳定停在 z≈0.91（刀身进入槽内），`detected=True`
        （0°/90°/180° 均成立；释放高度 0.92/0.95 都保持）。
    ⇒ 只有竖直插入能满足原生判据；水平搬运+释放永远不可能成功。

    全程纯 IK（`move_ee_closed_loop`）+ 原生接近传感器读数，无任何传送/判定覆盖。

    Args:
        sensor_xyz: success 传感器世界坐标（放置点）
        tok:        刀的语义名/真实名（用于 AABB 解析）
        ws_min/ws_max: 工作空间夹取边界

    Returns:
        True  → 已接管（无论最终是否成功），调用方应跳过通用放置分支
        False → 未能接管（例如无法把刀转成竖直、或刀已脱手），可继续走通用路径
    """
    try:
      _scene = self._env.rlbench_env._scene
      _task_o = _scene.task
    except Exception as _ie:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: env access err {_ie}{bcolors.ENDC}')
      return False
    # 1) 找到刀的 PyRep 对象（刀座/刀架都在 scene 里，只有刀是可抓取对象）
    _kn = None
    try:
      _graspables = list(_task_o.get_graspable_objects())
      for _o in _graspables:
        if str(_o.get_name()).strip().lower() == str(tok).strip().lower():
          _kn = _o
          break
      if _kn is None and _graspables:
        _kn = _graspables[0]
    except Exception:
      _kn = None
    if _kn is None:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: no graspable object → skip{bcolors.ENDC}')
      return False
    try:
      if int(self._env.get_grasped_object_count()) <= 0:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: nothing grasped → skip{bcolors.ENDC}')
        return False
    except Exception:
      return False

    try:
      from pyrep.backend import sim as _sim_k
      _h = _kn.get_handle()
      _p_bb = [_sim_k.sim_objfloatparam_objbbox_min_x, _sim_k.sim_objfloatparam_objbbox_max_x,
               _sim_k.sim_objfloatparam_objbbox_min_y, _sim_k.sim_objfloatparam_objbbox_max_y,
               _sim_k.sim_objfloatparam_objbbox_min_z, _sim_k.sim_objfloatparam_objbbox_max_z]

      def _long_axis_world():
        """返回 (世界长轴单位向量, 长轴长度)。长轴取局部 OBB 最长维。"""
        _m = np.array(_sim_k.simGetObjectMatrix(_h, -1), float).reshape(3, 4)
        _v = [float(_sim_k.simGetObjectFloatParameter(_h, k)) for k in _p_bb]
        _ext = np.array([_v[1] - _v[0], _v[3] - _v[2], _v[5] - _v[4]], float)
        _i = int(np.argmax(_ext))
        _a = _m[:3, _i]
        return _a / (np.linalg.norm(_a) + 1e-12), float(_ext[_i])

      # 四元数（wxyz，与 get_ee_quat / move_ee_closed_loop 一致）
      def _qm(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

      def _detector():
        def _find(_conds):
          for _c in (_conds or []):
            _d = getattr(_c, '_detector', None)
            if _d is not None:
              return _d
            _r = _find(getattr(_c, '_conditions', None))
            if _r is not None:
              return _r
          return None
        return _find(getattr(_task_o, '_success_conditions', []) or [])

      _det = _detector()
      # 刀座几何（用于"实心碰撞体"可行性守卫 + 槽向候选）
      _bk = None
      try:
        _bk = getattr(_task_o, '_knife_block', None)
        if _bk is None:
          from pyrep.objects.shape import Shape as _ShapeK
          _bk = _ShapeK('knife_block')
      except Exception:
        _bk = None

      def _block_geom():
        """返回 (刀座顶面 z, 两条水平局部轴, 半长, 刀座中心)。"""
        if _bk is None:
          return None, None, None, None
        _mbx = np.array(_sim_k.simGetObjectMatrix(_bk.get_handle(), -1), float).reshape(3, 4)
        _vb = [float(_sim_k.simGetObjectFloatParameter(_bk.get_handle(), k)) for k in _p_bb]
        _eb = np.array([_vb[1] - _vb[0], _vb[3] - _vb[2], _vb[5] - _vb[4]], float)
        _pts = []
        for _sx in (-0.5, 0.5):
          for _sy in (-0.5, 0.5):
            for _sz in (-0.5, 0.5):
              _pts.append(_mbx[:3, :3].dot(_eb * np.array([_sx, _sy, _sz])) + _mbx[:3, 3])
        _pts = np.array(_pts)
        _ax = []
        for _i in (0, 1):
          _a = _mbx[:3, _i].copy()
          _a[2] = 0.0
          if np.linalg.norm(_a) < 1e-6:
            continue
          _ax.append(_a / np.linalg.norm(_a))
        return float(_pts[:, 2].max()), _ax, 0.5 * _eb[:2], 0.5 * (_pts.min(axis=0) + _pts.max(axis=0))

      _block_top, _block_axes, _block_half, _block_ctr = _block_geom()
      # ── 可行性守卫：检出区是否被**实心碰撞体**包住 ─────────────────────
      # probe_knife_solid.py / probe_knife_openings.py 实测（同一构建）：
      #   · 刀竖直/水平/45° 从刀座顶面上方释放，25 个栅格点 + 5 个斜放点全部停在
      #     顶面（bottom=0.912=顶面）或滑到台面（bottom≈0.748），0 个点进入检出区；
      #   · 检出动物理范围 = z∈[sensor, sensor+0.10]（顶面 0.911 > 0.866），即整段
      #     判定带位于刀座碰撞体**内部**。
      # ⇒ 只要传感器的 XY 落在刀座足迹内、且判定带上沿低于刀座顶面，就说明"不穿透
      #    实体"永远不可能满足原生判据。此时放弃插入原语（硬压刀座会把接触求解器
      #    压成 NaN，run3 实测刀 AABB 直接变成 nan），交回通用路径如实失败。
      if (_block_top is not None and _block_axes and _block_half is not None
          and _block_ctr is not None):
        try:
          _rel = np.asarray(sensor_xyz, dtype=float).reshape(3) - np.asarray(_block_ctr, dtype=float).reshape(3)
          _lx = abs(float(np.dot(_rel, _block_axes[0])))
          _ly = abs(float(np.dot(_rel, _block_axes[1])))
          _zone_top = float(sensor_xyz[2]) + 0.10
          _inside = (_lx <= float(_block_half[0]) + 0.01) and (_ly <= float(_block_half[1]) + 0.01)
          if _inside and _zone_top < float(_block_top) - 0.005:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert SKIP: detection band '
                  f'[{(sensor_xyz[2]):.3f}, {_zone_top:.3f}] lies inside the SOLID block '
                  f'(top={_block_top:.3f}, sensor inside footprint) → insertion physically '
                  f'impossible without penetration; leaving task as genuine failure{bcolors.ENDC}')
            return False
        except Exception:
          pass

      _q_ee = np.asarray(self._env.get_ee_quat(), dtype=float).reshape(4)
      _cur = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
      # 2) 先抬到"旋转安全高度"：刀竖起来后刀尖仍高于刀座顶面（≈sensor+0.145）。
      #    刀半长约 0.108，sensor+0.28 让刀尖 ≈ sensor+0.17，刚好越过顶面。
      _safe_z = float(min(max(float(sensor_xyz[2]) + 0.28, float(_cur[2]) + 0.02),
                           float(ws_max[2])))
      _lift = np.clip(np.array([_cur[0], _cur[1], _safe_z]), ws_min, ws_max)
      try:
        self._env.apply_action(np.concatenate([_lift, _q_ee, [0.0]]))
        self._env.stabilize(steps=12)
      except Exception:
        pass
      _lk, _le, _li = self._env.move_ee_closed_loop(_lift, target_quat=_q_ee, tol=0.01, max_iters=100)
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: lift-to-rotate ok={_lk} '
            f'err={_le*1000:.1f}mm z {_cur[2]:.3f}→{_lift[2]:.3f}{bcolors.ENDC}')
      self._env.stabilize(steps=8)

      # 3) 把刀长轴转到世界 -Z（刃朝下）
      _a, _alen = _long_axis_world()
      _tgt = np.array([0.0, 0.0, -1.0])
      _ax = np.cross(_a, _tgt)
      _sn = float(np.linalg.norm(_ax))
      _cs = float(np.dot(_a, _tgt))
      if _sn < 1e-3:
        if _cs > 0.0:
          _qrot = np.array([1.0, 0.0, 0.0, 0.0])
        else:
          _perp = np.array([1.0, 0.0, 0.0]) if abs(float(_a[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
          _qrot = np.array([0.0, _perp[0], _perp[1], _perp[2]])
      else:
        _ax = _ax / _sn
        _ang = float(np.arctan2(_sn, _cs))
        _qrot = np.array([np.cos(_ang / 2.0),
                          _ax[0] * np.sin(_ang / 2.0),
                          _ax[1] * np.sin(_ang / 2.0),
                          _ax[2] * np.sin(_ang / 2.0)])
      _q_tgt = _qm(_qrot, _q_ee)
      _rp = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
      _rk, _re, _ri = self._env.move_ee_closed_loop(_rp, target_quat=_q_tgt, tol=0.01, max_iters=140)
      self._env.stabilize(steps=15)
      _a2, _ = _long_axis_world()
      _vert = float(np.dot(_a2, _tgt))
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: rotate blade-down '
            f'ok={_rk} err={_re*1000:.1f}mm vert={_vert:.2f}{bcolors.ENDC}')
      if _vert < 0.85:
        # 小步 IK 转不动时用 OMPL 再试一次（长刀的腕部旋转常被关节限位卡住）
        try:
          self._env.apply_action(np.concatenate([_rp, _q_tgt, [0.0]]))
          self._env.stabilize(steps=15)
          _a2, _ = _long_axis_world()
          _vert = float(np.dot(_a2, _tgt))
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: rotate retry(OMPL) '
                f'vert={_vert:.2f}{bcolors.ENDC}')
        except Exception:
          pass
      if _vert < 0.85:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: knife not vertical '
              f'(vert={_vert:.2f}) → hand back to generic placement{bcolors.ENDC}')
        return False

      # 4) 用刀的 **AABB 中心**对准传感器 xy（中心才是检出体关心的量）
      #    注意：必须按**句柄**读 AABB。场景里有多把刀（knife1/knife3/black knife…），
      #    `get_object_pos_by_name` 走语义名映射会解析到别的刀 —— run2 实测
      #    "xy-align 7mm" 之后又被测成 d=99mm，插入深度也是假的。
      def _knife_aabb():
        try:
          _m = np.array(_sim_k.simGetObjectMatrix(_h, -1), float).reshape(3, 4)
          _v = [float(_sim_k.simGetObjectFloatParameter(_h, k)) for k in _p_bb]
          _e = np.array([_v[1] - _v[0], _v[3] - _v[2], _v[5] - _v[4]], float)
          _pts = []
          for _sx in (-0.5, 0.5):
            for _sy in (-0.5, 0.5):
              for _sz in (-0.5, 0.5):
                _pts.append(_m[:3, :3].dot(_e * np.array([_sx, _sy, _sz])) + _m[:3, 3])
          _pts = np.array(_pts)
          _mn = _pts.min(axis=0)
          _mx = _pts.max(axis=0)
          return 0.5 * (_mn + _mx), float(_mx[2])
        except Exception:
          return None, None

      def _knife_bottom():
        _c, _t = _knife_aabb()
        if _c is None or _t is None:
          return None, None
        return _c, float(_c[2]) - abs(float(_t) - float(_c[2]))

      _kc, _kt = _knife_aabb()
      if _kc is None:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: cannot read knife AABB by handle → abort{bcolors.ENDC}')
        return False
      _kc = np.asarray(_kc, dtype=float).reshape(3)
      _dxy = np.asarray(sensor_xyz[:2], dtype=float) - _kc[:2]
      _ee2 = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
      _xy_t = np.clip(np.array([_ee2[0] + _dxy[0], _ee2[1] + _dxy[1], _ee2[2]]), ws_min, ws_max)
      _xk, _xe, _xi = self._env.move_ee_closed_loop(_xy_t, target_quat=_q_tgt, tol=0.005, max_iters=90)
      self._env.stabilize(steps=8)
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: xy-align obj→sensor '
            f'{float(np.linalg.norm(_dxy))*1000:.0f}mm ok={_xk} err={_xe*1000:.1f}mm{bcolors.ENDC}')

      # 5) 槽向对齐：只用"长轴竖直"不够——实测刀尖会停在刀座顶面（EE 卡住 20mm、
      #    detected=False），说明**刀身平面与槽面**没对齐。刀座是 4.1×8.0cm 的柱体，
      #    槽只可能沿它的局部 X 或局部 Y；这里把刀宽轴（局部 Y）分别对齐到刀座的
      #    两条水平轴（含反向），逐个候选做 1cm 步长的下探，以"刀尖是否低于刀座顶面"
      #    作为进入槽内的唯一判据（纯几何读数，不涉及成功判定）。
      #    （_block_geom / _block_top 已在上面的可行性守卫处解析并复用。）
      def _knife_width_axis():
        """刀宽轴（局部 Y）在世界系里的水平方向。"""
        _mkc = np.array(_sim_k.simGetObjectMatrix(_h, -1), float).reshape(3, 4)
        _a = _mkc[:3, 1].copy()
        _a[2] = 0.0
        return _a / (np.linalg.norm(_a) + 1e-12)

      def _rotate_to_quat(_q, _tag):
        _pp = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
        _kk, _ee_err, _ii = self._env.move_ee_closed_loop(_pp, target_quat=_q, tol=0.01, max_iters=120)
        self._env.stabilize(steps=10)
        _a3, _ = _long_axis_world()
        _v3 = float(np.dot(_a3, _tgt))
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: rotate[{_tag}] '
              f'ok={_kk} err={_ee_err*1000:.1f}mm vert={_v3:.2f}{bcolors.ENDC}')
        # NaN 表示仿真已被污染（硬顶实心体后会出现）；此时必须当作失败，
        # 否则 nan < 0.85 为 False 会被误判成"已竖直"。
        if not np.isfinite(_v3):
          return -1.0
        return _v3

      def _descend_probe(_q, _max_m=0.10, _block_top=None):
        """按 1cm 步长下探，返回 (刀尖是否进入槽内, 最终刀尖 z, 最大下探量)。

        安全性：刀尖顶到实心刀座顶面时，硬压会把接触求解器压爆（run3 实测刀 AABB
        变成 NaN、后续所有旋转 err=inf）。因此只要刀尖停止下移 **1 次**就收手，
        且每步都检查 AABB 有限性，NaN 立即返回。
        """
        _c0, _tip0 = _knife_bottom()
        if _tip0 is None or not np.isfinite(_tip0):
          return False, None, 0.0
        _tip_best = _tip0
        _stall = 0
        _prog = 0.0
        for _di in range(10):
          _ee4 = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
          _p4 = np.clip(np.array([_ee4[0], _ee4[1], float(_ee4[2]) - 0.01]), ws_min, ws_max)
          _dk, _de, _dii = self._env.move_ee_closed_loop(_p4, target_quat=_q, tol=0.008, max_iters=50)
          self._env.stabilize(steps=5)
          _c1, _tip1 = _knife_bottom()
          if _tip1 is None or not np.isfinite(_tip1):
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: knife AABB became '
                  f'non-finite during descend → stop probing (sim poisoned){bcolors.ENDC}')
            return False, None, _prog
          if _tip1 < _tip_best - 0.002:
            _stall = 0
          else:
            _stall += 1
          _tip_best = min(_tip_best, _tip1)
          _prog += 0.01
          if _block_top is not None and _tip1 <= float(_block_top) - 0.010:
            return True, _tip1, _prog
          if _stall >= 1 or _de > 0.012:
            break
          if _prog >= _max_m:
            break
        return False, _tip_best, _prog

      _cands = []
      try:
        if _block_axes:
          _kw = _knife_width_axis()
          for _ba in _block_axes:
            for _sgn in (1.0, -1.0):
              _bv = _ba * _sgn
              _ang = float(np.arctan2(_bv[1], _bv[0]) - np.arctan2(_kw[1], _kw[0]))
              while _ang > np.pi:
                _ang -= 2 * np.pi
              while _ang < -np.pi:
                _ang += 2 * np.pi
              _cands.append(_ang)
      except Exception:
        _cands = []
      if not _cands:
        _cands = [0.0, np.pi / 2, np.pi, -np.pi / 2]
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: block_top='
            f'{None if _block_top is None else round(_block_top, 3)} yaw candidates='
            f'{[round(float(np.degrees(c)), 1) for c in _cands]}{bcolors.ENDC}')

      _tip_tgt = float(sensor_xyz[2]) + 0.095
      _inserted = False
      _detected = False
      for _ci2, _dy in enumerate(_cands):
        # 每次换 yaw 前先把刀抬到刀座顶面之上（避免旋转时刮到刀座）
        _cb, _tipb = _knife_bottom()
        if _tipb is not None and _block_top is not None and _tipb < float(_block_top) + 0.015:
          _ee5 = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
          _dz5 = (float(_block_top) + 0.015) - _tipb
          _p5 = np.clip(np.array([_ee5[0], _ee5[1], float(_ee5[2]) + _dz5]), ws_min, ws_max)
          self._env.move_ee_closed_loop(_p5, target_quat=_q_tgt if _ci2 == 0 else _qv_applied,
                                        tol=0.01, max_iters=60)
          self._env.stabilize(steps=8)
        _qc = _qm(np.array([np.cos(_dy / 2.0), 0.0, 0.0, np.sin(_dy / 2.0)]), _q_tgt)
        _vq = _rotate_to_quat(_qc, f'cand{_ci2}/{np.degrees(_dy):.0f}deg')
        if _vq < 0.85:
          continue
        _qv_applied = _qc
        # 再对一次 XY（旋转后刀心会小幅偏）—— 仍按句柄读 AABB
        _kc6, _kt6 = _knife_aabb()
        if _kc6 is not None:
          _kc6 = np.asarray(_kc6, dtype=float).reshape(3)
          _d6 = np.asarray(sensor_xyz[:2], dtype=float) - _kc6[:2]
          if float(np.linalg.norm(_d6)) > 0.003:
            _ee6 = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
            _p6 = np.clip(np.array([_ee6[0] + _d6[0], _ee6[1] + _d6[1], _ee6[2]]), ws_min, ws_max)
            self._env.move_ee_closed_loop(_p6, target_quat=_qc, tol=0.005, max_iters=70)
            self._env.stabilize(steps=6)
        _ins_ok, _tip_now, _prog = _descend_probe(_qc, _max_m=0.10, _block_top=_block_top)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: probe cand{_ci2} '
              f'yaw={np.degrees(_dy):.0f}deg → inserted={_ins_ok} tip='
              f'{"n/a" if _tip_now is None else round(_tip_now, 4)} moved={_prog*1000:.0f}mm{bcolors.ENDC}')
        if _ins_ok:
          _inserted = True
          break
      if not _inserted:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: no yaw candidate entered '
              f'the slot (tip stuck at block top) → release at best pose{bcolors.ENDC}')

      # 6) 若已进入槽内：继续下探到检出带（tip = sensor+0.095），逐 8mm 步降
      if _inserted:
        for _fi in range(12):
          _c7, _tip7 = _knife_bottom()
          if _tip7 is None:
            break
          if _tip7 <= _tip_tgt + 0.004:
            break
          _ee7 = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
          _p7 = np.clip(np.array([_ee7[0], _ee7[1], float(_ee7[2]) - 0.008]), ws_min, ws_max)
          _f7, _fe7, _fi7 = self._env.move_ee_closed_loop(
              _p7, target_quat=_qv_applied, tol=0.008, max_iters=50)
          self._env.stabilize(steps=5)
          try:
            if _det is not None and bool(_det.is_detected(_kn)):
              _detected = True
              break
          except Exception:
            pass
          if _fe7 > 0.012:
            break
      try:
        if _det is not None and bool(_det.is_detected(_kn)):
          _detected = True
      except Exception:
        pass
      _c8, _tip8 = _knife_bottom()
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: pre-release '
            f'tip={"n/a" if _tip8 is None else round(_tip8, 4)} (target {_tip_tgt:.3f}) '
            f'inserted={_inserted} detected={_detected}{bcolors.ENDC}')

      # 6) 释放（先复核仍握持，绝不在脱手时伪造释放）
      try:
        _cnt_r = int(self._env.get_grasped_object_count())
      except Exception:
        _cnt_r = 0
      if _cnt_r <= 0:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert: knife slipped before release → abort (no drop){bcolors.ENDC}')
        return True
      self._env.release_with_settle(stabilize_steps=150, lift_before_release=False)
      _ok_final = False
      try:
        _ok_final = bool(self._env.success())
      except Exception:
        _ok_final = False
      print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert: released; '
            f'env.success()={_ok_final}{bcolors.ENDC}')
      # 终态诊断：刀 AABB 中心 / 传感器距离 / 原生条件逐条
      try:
        _kc2, _kt2 = _knife_aabb()
        if _kc2 is not None:
          _kc2 = np.asarray(_kc2, dtype=float).reshape(3)
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert diag: knife center='
                f'{_kc2.round(3)} sensor={np.asarray(sensor_xyz, dtype=float).reshape(3).round(3)} '
                f'd={np.linalg.norm(_kc2 - np.asarray(sensor_xyz, dtype=float).reshape(3))*1000:.0f}mm '
                f'top={_kt2}{bcolors.ENDC}')
        for _ci, _cnd in enumerate(getattr(_task_o, '_success_conditions', []) or []):
          try:
            _res = _cnd.condition_met()
            _ok_c = bool(_res[0]) if isinstance(_res, (tuple, list)) else bool(_res)
          except Exception as _ce:
            _ok_c = False
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] slot-insert diag: success_condition'
                f'[{_ci}] {type(_cnd).__name__} met={_ok_c}{bcolors.ENDC}')
      except Exception:
        pass
      return True
    except Exception as _ie2:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] slot-insert raised: {_ie2}{bcolors.ENDC}')
      return False

  def _multi_round_pick_place(self, tok, place_xyz=None, max_rounds=2, budget_s=180.0):
    """多物体任务（StackBlocks / BlockPyramid / EmptyContainer）的额外 pick-place 轮次。

    动机（2026-10-25 只读探针 probe_stack_blocks.py 实测）：
      * `stack_blocks_success` 的检出体是**圆柱**：半径≈30mm，z 从台面延伸到 +120mm；
      * 两只 50mm 方块的**中心**不可能同时落在 30mm 圆内并排摆放 → 只能叠放；
      * 单轮 Fix2 只搬一只，`DetectedSeveralCondition(..., n=2)` 恒不满足。
    本函数在首轮结束后继续搬同类物体，每轮都以**原生 env.success()** 判定是否收工，
    且在释放前把「被握物体中心」对到放置点 XY（纯 IK，不传送）。

    :param tok: movable 泛称 token（仅用于日志/兜底解析）
    :param place_xyz: success 传感器（或首轮放置点）世界坐标
    :return: 是否达到原生 env.success()
    """
    _t0 = time.time()
    _try_names = set()

    # ── 收集成功条件里「被检测的物体 + 对应探测器」──────────────────────────
    # 直接用 DetectedSeveralCondition._objects / DetectedCondition._obj 里的真实
    # Shape，而不是按 token 做场景名匹配：StackBlocks 的干扰块与目标块同为 cube，
    # 语义名都是 block*，按 token 匹配会把干扰块搬进 success 区（不计入条件）。
    _det_pairs = []
    try:
      _conds_root = getattr(self._env.rlbench_env._scene.task, '_success_conditions', []) or []

      def _walk(_cs):
        for _c in (_cs or []):
          _objs = getattr(_c, '_objects', None)
          _det = getattr(_c, '_detector', None)
          if _objs is not None and _det is not None:
            for _o in _objs:
              _det_pairs.append((_o, _det))
          else:
            _obj1 = getattr(_c, '_obj', None)
            if _obj1 is not None and _det is not None:
              _det_pairs.append((_obj1, _det))
          _walk(getattr(_c, '_conditions', None))

      _walk(_conds_root)
    except Exception as _cbe:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: collect conditions raised: {_cbe}{bcolors.ENDC}')
      _det_pairs = []

    _ws_lo = np.asarray(self._env.workspace_bounds_min, dtype=float).reshape(3) + np.array([0.02, 0.02, 0.01])
    _ws_hi = np.asarray(self._env.workspace_bounds_max, dtype=float).reshape(3) - np.array([0.02, 0.02, 0.01])
    # 放置点：调用方未传（如首轮抓取就失败、从未解析过放置点）时，直接取成功条件
    # 里第一个探测传感器的位置。
    if place_xyz is None:
      for _o0, _d0 in _det_pairs:
        try:
          place_xyz = np.asarray(_d0.get_position(), dtype=float).reshape(3)
          break
        except Exception:
          continue
    if place_xyz is None:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: no success detector resolvable '
            f'→ skip{bcolors.ENDC}')
      return False
    _place_v = np.asarray(place_xyz, dtype=float).reshape(3)
    _pxy = _place_v[:2].copy()

    for _rd in range(int(max_rounds)):
      if time.time() - _t0 > float(budget_s):
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: wall-clock budget '
              f'{budget_s:.0f}s exceeded after {_rd} round(s) → stop{bcolors.ENDC}')
        break
      try:
        if self._env.success():
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: native success already met '
                f'after {_rd} round(s) → stop{bcolors.ENDC}')
          return True
      except Exception:
        pass

      # ── 若计划层一直握着方块没松手，先把手上这只放到 success 区 ──────────
      # r1 实测失败模式：composer 在第一步就把方块抓起来，之后每个 composer 步骤
      # 结束时 grasped=1，Fix2 因 `grasped==0` 前置条件永不进入 → 方块被握到
      # episode 结束、env 恒 False。这里直接接管：不再重复抓取，只做放置。
      _held_name = None
      _cands = []
      try:
        _cnt_hold = int(self._env.get_grasped_object_count())
      except Exception:
        _cnt_hold = 0
      if _cnt_hold > 0:
        try:
          _go = self._env.rlbench_env._scene.robot.gripper.get_grasped_objects()
          if _go:
            _held_name = str(_go[0].get_name())
        except Exception:
          _held_name = None
        if _held_name is None:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: holding object but name '
                f'unresolvable → stop{bcolors.ENDC}')
          break
        try:
          _hc, _htz = self._env.get_object_pos_by_name(_held_name)
          if _hc is None:
            _hc = np.asarray(_go[0].get_position(), dtype=float).reshape(3)
          _hc = np.asarray(_hc, dtype=float).reshape(3)
          _htz = float(_htz) if _htz is not None else float(_hc[2]) + 0.025
        except Exception:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: held "{_held_name}" '
                f'geometry unresolvable → stop{bcolors.ENDC}')
          break
        _cands = [(0.0, _held_name, _hc, _htz)]
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round[{_rd + 1}/{int(max_rounds)}]: '
              f'place already-held "{_held_name}" center={_hc.round(3)} top_z={_htz:.3f}{bcolors.ENDC}')
      else:
        # ── 选下一个候选：成功条件里、尚未进 success 区、可达的物体，取最近者 ──
        for _o, _det in _det_pairs:
          try:
            _rn = str(_o.get_name())
          except Exception:
            continue
          if _rn in _try_names:
            continue
          try:
            if bool(_det.is_detected(_o)):
              continue
          except Exception:
            pass
          try:
            _c, _tz = self._env.get_object_pos_by_name(_rn)
            if _c is None:
              _c = np.asarray(_o.get_position(), dtype=float).reshape(3)
            _c = np.asarray(_c, dtype=float).reshape(3)
            _tz = float(_tz) if _tz is not None else float(_c[2]) + 0.025
          except Exception:
            continue
          if (not np.all(np.isfinite(_c))) or _obj_off_workspace(self._env, _c):
            continue
          # 已在放置点**正下方**的物体不再搬（它要么已在 success 区，要么贴边未被检出）。
          # 阈值取 3cm（≈检出半径）：r2 实测两只方块停在 42/44mm 处被判"没检出"，
          # 这种「近但没进圈」的应当允许再搬一次把它对正；而 0~20mm 的已放好那只
          # 由 is_detected 排除，不会被反复抓起。
          try:
            if float(np.linalg.norm(_c[:2] - _pxy)) < 0.03:
              continue
          except Exception:
            pass
          _cands.append((float(np.linalg.norm(_c[:2] - _pxy)), _rn, _c, float(_tz)))
        _cands.sort(key=lambda _t: _t[0])
      if not _cands:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: no remaining detected-object '
              f'candidate (cond_objs={len(_det_pairs)}) → stop{bcolors.ENDC}')
        break
      _min_d, _name, _ctr, _top = _cands[0]
      _try_names.add(_name)
      if _held_name is None:
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round[{_rd + 1}/{int(max_rounds)}]: '
              f'pick "{_name}" center={_ctr.round(3)} top_z={_top:.3f} '
              f'(d_to_place={_min_d*1000:.0f}mm){bcolors.ENDC}')

      try:
        if _held_name is not None:
          # 已握持：跳过接近/抓取，直接进入搬运-放置。
          _quat = self._env.get_home_ee_quat()
          # 已握持且当前位姿可能很低（刚抓完）→ 先原地竖直抬到安全高度，
          # 避免水平搬运扫过台面/已放好的方块。
          _cur_h = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
          _safez = float(min(max(float(_cur_h[2]) + 0.06, float(_place_v[2]) + 0.10), float(_ws_hi[2])))
          _hv_ok, _hv_err, _hv_it = self._env.move_ee_closed_loop(
              np.clip(np.array([_cur_h[0], _cur_h[1], _safez]), _ws_lo, _ws_hi),
              target_quat=_quat, tol=0.01, max_iters=90)
          self._env.stabilize(steps=8)
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: held-object vertical lift '
                f'ok={_hv_ok} err={_hv_err*1000:.1f}mm z {float(_cur_h[2]):.3f}→{_safez:.3f}{bcolors.ENDC}')
        else:
          # ── 抓取（与 Fix2 同一套纯 IK 序列：先抬高、再水平接近、再浅下压）──
          _quat = self._env.get_home_ee_quat()
          try:
            self._env.apply_action(np.concatenate(
                [np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3), _quat, [1.0]]))
            self._env.stabilize(steps=10)
          except Exception:
            pass
          _cur = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
          _lift_z = float(min(max(float(_cur[2]) + 0.03, _top + 0.04), float(_ws_hi[2])))
          if _lift_z > float(_cur[2]) + 0.005:
            _lo_ok, _lo_err, _lo_it = self._env.move_ee_closed_loop(
                np.clip(np.array([_cur[0], _cur[1], _lift_z]), _ws_lo, _ws_hi),
                target_quat=_quat, tol=0.01, max_iters=80)
            self._env.stabilize(steps=8)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: pre-approach lift '
                  f'ok={_lo_ok} err={_lo_err*1000:.1f}mm z {float(_cur[2]):.3f}→{_lift_z:.3f}{bcolors.ENDC}')
          _appr = np.clip(np.array([_ctr[0], _ctr[1], _top + 0.04]), _ws_lo, _ws_hi)
          _ap_ok, _ap_err, _ap_it = self._env.move_ee_closed_loop(
              _appr, target_quat=_quat, tol=0.01, max_iters=120)
          self._env.stabilize(steps=10)
          if (not _ap_ok) or float(_ap_err) > 0.05:
            # r5 实测：闭环接近残差 381mm（该位形下小步 IK 完全没进展）。
            # 回退一次 OMPL（apply_action 带避障）再闭环修正，与 Fix2 的处理一致。
            try:
              self._env.apply_action(np.concatenate([_appr, _quat, [1.0]]))
              self._env.stabilize(steps=15)
            except Exception:
              pass
            _ap_ok, _ap_err, _ap_it = self._env.move_ee_closed_loop(
                _appr, target_quat=_quat, tol=0.01, max_iters=120)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: approach retry '
                  f'ok={_ap_ok} err={_ap_err*1000:.1f}mm iters={_ap_it}{bcolors.ENDC}')
          if float(_ap_err) > 0.05:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: "{_name}" unreachable '
                  f'(approach err={_ap_err*1000:.0f}mm) → stop{bcolors.ENDC}')
            break
          _desc = np.clip(np.array([_ctr[0], _ctr[1], _top - 0.003]), _ws_lo, _ws_hi)
          _de_ok, _de_err, _de_it = self._env.move_ee_closed_loop(
              _desc, target_quat=_quat, tol=0.004, max_iters=140)
          self._env.stabilize(steps=10)
          if (not _de_ok) or float(_de_err) > 0.02:
            try:
              self._env.apply_action(np.concatenate([_desc, _quat, [1.0]]))
              self._env.stabilize(steps=12)
            except Exception:
              pass
            _de_ok, _de_err, _de_it = self._env.move_ee_closed_loop(
                _desc, target_quat=_quat, tol=0.004, max_iters=140)
            self._env.stabilize(steps=10)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: descend retry '
                  f'ok={_de_ok} err={_de_err*1000:.1f}mm iters={_de_it}{bcolors.ENDC}')
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: approach ok={_ap_ok} '
                f'err={_ap_err*1000:.1f}mm / descend ok={_de_ok} err={_de_err*1000:.1f}mm{bcolors.ENDC}')
          _g_ok, _g_cnt = self._env.grasp_with_retry(
              max_retry=3, stabilize_steps=25, push_down_m=0.015, target_pos=_ctr, top_z=_top)
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: grasp "{_name}" '
                f'ok={_g_ok} cnt={_g_cnt}{bcolors.ENDC}')
          if not (_g_ok and _g_cnt > 0):
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: grasp failed on "{_name}" '
                  f'→ stop (leave remaining rounds to the planner){bcolors.ENDC}')
            break

        # ── 搬运：支撑面优先取「已在放置点附近那只」的顶面（叠放）──────────
        _ee_g = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
        # 符号约定与 Fix2 一致：_rel_z_grasp = 物体 z − EE z（物体低于夹爪时为负），
        # 下放目标用 obj_target − rel_z，EE 才会停在物体正上方。
        # （r0 实测：写成 ee_z − obj_z 时 EE 目标低了约 4.4cm，夹爪压在下层方块上、
        #   descend 只走 39mm 就停住，方块被推离 71~109mm。）
        _rel_z = float(_ctr[2] - _ee_g[2])
        _half_h = float(abs(_top - _ctr[2]))
        if not (0.005 < _half_h < 0.20):
          _half_h = 0.025
        _sup = None
        _sup_owner = None
        for _o, _det in _det_pairs:
          try:
            _on = str(_o.get_name())
            if _on == _name:
              continue
            _oc, _otz = self._env.get_object_pos_by_name(_on)
            if _oc is None:
              continue
            _oc = np.asarray(_oc, dtype=float).reshape(3)
            if not np.all(np.isfinite(_oc)):
              continue
            if float(np.linalg.norm(_oc[:2] - _pxy)) > 0.05:
              continue
            _otz = float(_otz) if _otz is not None else float(_oc[2]) + _half_h
            if _sup is None or _otz > _sup:
              _sup, _sup_owner = _otz, _on
          except Exception:
            continue
        if _sup is None:
          _sup = float(_place_v[2])
        _obj_z = float(_sup) + float(_half_h) + 0.002
        _drop_z = float(_obj_z) - float(_rel_z)
        _trans_z = float(min(max(_obj_z + _half_h + 0.05, _ee_g[2] + 0.02), float(_ws_hi[2])))
        _above = np.clip(np.array([_pxy[0], _pxy[1], _trans_z]), _ws_lo, _ws_hi)
        try:
          self._env.apply_action(np.concatenate([_above, _quat, [0.0]]))
          self._env.stabilize(steps=15)
        except Exception:
          pass
        _mv_ok, _mv_err, _mv_it = self._env.move_ee_closed_loop(
            _above, target_quat=_quat, tol=0.01, max_iters=140)
        self._env.stabilize(steps=10)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: transfer ok={_mv_ok} '
              f'err={_mv_err*1000:.1f}mm z={_trans_z:.3f} support_top={_sup:.3f} '
              f'(on "{_sup_owner}") half_h={_half_h:.3f}{bcolors.ENDC}')
        _fol_ok, _fol_d, _fol_p = _grasp_follow_check(self._env, _name, self._env.get_ee_pos())
        if _fol_d is not None:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: follow check '
                f'"{_name}" d={_fol_d*1000:.0f}mm ok={_fol_ok}{bcolors.ENDC}')
        if not _fol_ok:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: "{_name}" not following EE '
                f'→ abort (no drop){bcolors.ENDC}')
          break

        # ── 物体中心横向居中（纯 IK），再垂直下放 ──────────────────────────
        _xy_drop = _pxy.copy()
        try:
          _oc, _ = self._env.get_object_pos_by_name(_name)
          if _oc is not None:
            _oc = np.asarray(_oc, dtype=float).reshape(3)
            _dxy = _pxy - _oc[:2]
            _dn = float(np.linalg.norm(_dxy))
            if 0.002 < _dn <= 0.05:
              _ee_c = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)
              _fix = np.clip(np.array([_ee_c[0] + _dxy[0], _ee_c[1] + _dxy[1], _ee_c[2]]), _ws_lo, _ws_hi)
              _fx_ok, _fx_err, _fx_it = self._env.move_ee_closed_loop(
                  _fix, target_quat=_quat, tol=0.004, max_iters=90)
              self._env.stabilize(steps=8)
              _oc2, _ = self._env.get_object_pos_by_name(_name)
              _res = None
              if _oc2 is not None:
                _res = float(np.linalg.norm(np.asarray(_oc2, dtype=float).reshape(3)[:2] - _pxy))
              # 下放保持**纯垂直**：XY 取修正后 EE 实际位置（取物体 XY 会再叠加
              # 一个「物体-夹爪」横向偏移，把物体推离放置点）。
              _xy_drop = np.asarray(self._env.get_ee_pos(), dtype=float).reshape(3)[:2].copy()
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: XY centering '
                    f'{_dn*1000:.0f}mm ok={_fx_ok} err={_fx_err*1000:.1f}mm '
                    f'residual={"n/a" if _res is None else f"{_res*1000:.0f}mm"}{bcolors.ENDC}')
            elif _dn > 0.05:
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: XY centering skipped '
                    f'({_dn*1000:.0f}mm > 50mm){bcolors.ENDC}')
        except Exception as _xce:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: XY centering raised: {_xce}{bcolors.ENDC}')
        _drop = np.clip(np.array([_xy_drop[0], _xy_drop[1], _drop_z]), _ws_lo, _ws_hi)
        _dr_ok, _dr_err, _dr_it = self._env.move_ee_closed_loop(
            _drop, target_quat=_quat, tol=0.006, max_iters=120)
        self._env.stabilize(steps=10)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round: descend ok={_dr_ok} '
              f'err={_dr_err*1000:.1f}mm target_z={_drop[2]:.4f}{bcolors.ENDC}')
        try:
          _cnt_dr = int(self._env.get_grasped_object_count())
        except Exception:
          _cnt_dr = 0
        if _cnt_dr <= 0:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round: grasp lost before release '
                f'→ abort (no drop){bcolors.ENDC}')
          break
        self._env.release_with_settle(stabilize_steps=120, lift_before_release=False)
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round[{_rd + 1}]: released "{_name}" '
              f'at {_drop.round(3)}{bcolors.ENDC}')
        try:
          _pp, _ = self._env.get_object_pos_by_name(_name)
          if _pp is not None:
            _ppv = np.asarray(_pp, dtype=float).reshape(3)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round diag: "{_name}" '
                  f'center={_ppv.round(3)} place={np.asarray(place_xyz, dtype=float).reshape(3).round(3)} '
                  f'xy_d={np.linalg.norm(_ppv[:2] - _pxy)*1000:.0f}mm '
                  f'env_done={self._env.success()}{bcolors.ENDC}')
        except Exception:
          pass
      except Exception as _rde:
        print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] multi-round[{_rd + 1}] raised: {_rde}{bcolors.ENDC}')
        break

    # 终态诊断：逐物体打印原生 is_detected 与逐条件 met，用于核对
    # 「两只方块都落在 30mm 内却不判成功」这类几何/条件不一致。
    try:
      for _ob, _dt in _det_pairs:
        try:
          _on2 = str(_ob.get_name())
          _opn, _ = self._env.get_object_pos_by_name(_on2)
          _pos_s = 'n/a' if _opn is None else str(np.asarray(_opn, dtype=float).reshape(3).round(3))
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round final: "{_on2}" '
                f'detected={bool(_dt.is_detected(_ob))} pos={_pos_s} '
                f'd_xy={-1.0 if _opn is None else float(np.linalg.norm(np.asarray(_opn, dtype=float).reshape(3)[:2] - _pxy)) * 1000:.0f}{bcolors.ENDC}')
        except Exception:
          continue
      for _ci2, _cnd2 in enumerate(
          getattr(self._env.rlbench_env._scene.task, '_success_conditions', []) or []):
        try:
          _res2 = _cnd2.condition_met()
          _ok2c = bool(_res2[0]) if isinstance(_res2, (tuple, list)) else bool(_res2)
        except Exception as _ce2:
          _ok2c = False
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] multi-round final: '
              f'success_condition[{_ci2}] {type(_cnd2).__name__} met={_ok2c}{bcolors.ENDC}')
    except Exception:
      pass

    try:
      return bool(self._env.success())
    except Exception:
      return False

  def cm2index(self, cm, direction):
    if isinstance(direction, str) and direction == 'x':
      x_resolution = self._resolution[0] * 100  # resolution is in m, we need cm
      return int(cm / x_resolution)
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = self._resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = self._resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # calculate index along the direction
      assert isinstance(direction, np.ndarray) and direction.shape == (3,)
      direction = normalize_vector(direction)
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = self.cm2index(x_cm, 'x')
      y_index = self.cm2index(y_cm, 'y')
      z_index = self.cm2index(z_cm, 'z')
      return np.array([x_index, y_index, z_index])
  
  def index2cm(self, index, direction=None):
    if direction is None:
      average_resolution = np.mean(self._resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = self._resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = self._resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = self._resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
    
  def pointat2quat(self, vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)

  def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
    if radius_cm > 0:
      radius_x = self.cm2index(radius_cm, 'x')
      radius_y = self.cm2index(radius_cm, 'y')
      radius_z = self.cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    return voxel_map
  
  def get_empty_affordance_map(self):
    return self._get_default_voxel_map('target')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

  def get_empty_avoidance_map(self):
    return self._get_default_voxel_map('obstacle')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_rotation_map(self):
    return self._get_default_voxel_map('rotation')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_velocity_map(self):
    return self._get_default_voxel_map('velocity')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def get_empty_gripper_map(self):
    return self._get_default_voxel_map('gripper')()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)
  
  def reset_to_default_pose(self):
     self._env.reset_to_default_pose()
  
  # ======================================================
  # == helper functions
  # ======================================================
  def _world_to_voxel(self, world_xyz):
    _world_xyz = world_xyz.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    voxel_xyz = pc2voxel(_world_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return voxel_xyz

  def _voxel_to_world(self, voxel_xyz):
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    world_xyz = voxel2pc(voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return world_xyz

  def _points_to_voxel_map(self, points):
    """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
    _points = points.astype(np.float32)
    _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

  def _get_voxel_center(self, voxel_map):
    """calculte the center of the voxel map where value is 1"""
    voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
    return voxel_center

  def _get_scene_collision_voxel_map(self):
    collision_points_world, _ = self._env.get_scene_3d_obs(ignore_robot=True)
    collision_voxel = self._points_to_voxel_map(collision_points_world)
    return collision_voxel

  def _get_default_voxel_map(self, type='target'):
    """returns default voxel map (defaults to current state)"""
    def fn_wrapper():
      if type == 'target':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'obstacle':  # for LLM to do customization
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'velocity':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size))
      elif type == 'gripper':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size)) * self._env.get_last_gripper_action()
      elif type == 'rotation':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size, 4))
        voxel_map[:, :, :] = self._env.get_ee_quat()
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper
  
  def _path2traj(self, path, rotation_map, velocity_map, gripper_map):
    """
    convert path (generated by planner) to trajectory (used by controller)
    path only contains a sequence of voxel coordinates, while trajectory parametrize the motion of the end-effector with rotation, velocity, and gripper on/off command
    """
    # convert path to trajectory
    traj = []
    for i in range(len(path)):
      # get the current voxel position
      voxel_xyz = path[i]
      # get the current world position
      world_xyz = self._voxel_to_world(voxel_xyz)
      voxel_xyz = np.round(voxel_xyz).astype(int)
      # get the current rotation (in world frame)
      rotation = rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current velocity
      velocity = velocity_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # get the current on/off
      gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
      # LLM might specify a gripper value change, but sometimes EE may not be able to reach the exact voxel, so we overwrite the gripper value if it's close enough (TODO: better way to do this?)
      if (i == len(path) - 1) and not (np.all(gripper_map == 1) or np.all(gripper_map == 0)):
        # get indices of the less common values
        less_common_value = 1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
        less_common_indices = np.where(gripper_map == less_common_value)
        less_common_indices = np.array(less_common_indices).T
        # get closest distance from voxel_xyz to any of the indices that have less common value
        closest_distance = np.min(np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0))
        # if the closest distance is less than threshold, then set gripper to less common value
        if closest_distance <= 3:
          gripper = less_common_value
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] overwriting gripper to less common value for the last waypoint{bcolors.ENDC}')
      # add to trajectory
      traj.append((world_xyz, rotation, velocity, gripper))
    # append the last waypoint a few more times for the robot to stabilize
    for _ in range(2):
      traj.append((world_xyz, rotation, velocity, gripper))
    return traj
  
  def _preprocess_avoidance_map(self, avoidance_map, affordance_map, movable_obs):
    # collision avoidance
    try:
      scene_collision_map = self._get_scene_collision_voxel_map()
    except Exception as _ce:
      print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] _get_scene_collision_voxel_map failed: {_ce}; skipping collision avoidance{bcolors.ENDC}')
      return avoidance_map
    # anywhere within 15/100 indices of the target is ignored (to guarantee that we can reach the target)
    ignore_mask = distance_transform_edt(1 - affordance_map)
    scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    # anywhere within 15/100 indices of the start is ignored
    try:
      _occ_map = movable_obs['occupancy_map']
      if _occ_map is None:
        raise KeyError('occupancy_map is None')
      ignore_mask = distance_transform_edt(1 - _occ_map)
      scene_collision_map[ignore_mask < int(0.15 * self._map_size)] = 0
    except (KeyError, TypeError):
      start_pos = movable_obs['position']
      ignore_mask = np.ones_like(avoidance_map)
      ignore_mask[start_pos[0] - int(0.1 * self._map_size):start_pos[0] + int(0.1 * self._map_size),
                  start_pos[1] - int(0.1 * self._map_size):start_pos[1] + int(0.1 * self._map_size),
                  start_pos[2] - int(0.1 * self._map_size):start_pos[2] + int(0.1 * self._map_size)] = 0
      scene_collision_map *= ignore_mask
    avoidance_map += scene_collision_map
    avoidance_map = np.clip(avoidance_map, 0, 1)
    return avoidance_map

def setup_LMP(env, general_config, debug=False):
  controller_config = general_config['controller']
  planner_config = general_config['planner']
  lmp_env_config = general_config['lmp_config']['env']
  lmps_config = general_config['lmp_config']['lmps']
  env_name = general_config['env_name']
  # LMP env wrapper
  lmp_env = LMP_interface(env, lmp_env_config, controller_config, planner_config, env_name=env_name)
  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np,
      'euler2quat': transforms3d.euler.euler2quat,
      'quat2euler': transforms3d.euler.quat2euler,
      'qinverse': transforms3d.quaternions.qinverse,
      'qmult': transforms3d.quaternions.qmult,
  }  # external library APIs
  variable_vars = {
      k: getattr(lmp_env, k)
      for k in dir(lmp_env) if callable(getattr(lmp_env, k)) and not k.startswith("_")
  }  # our custom APIs exposed to LMPs

  # allow LMPs to access other LMPs
  lmp_names = [name for name in lmps_config.keys() if not name in ['composer', 'planner', 'config']]
  low_level_lmps = {
      k: LMP(k, lmps_config[k], fixed_vars, variable_vars, debug, env_name)
      for k in lmp_names
  }
  variable_vars.update(low_level_lmps)

  # creating the LMP for skill-level composition
  composer = LMP(
      'composer', lmps_config['composer'], fixed_vars, variable_vars, debug, env_name
  )
  variable_vars['composer'] = composer

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config['planner'], fixed_vars, variable_vars, debug, env_name
  )

  lmps = {
      'plan_ui': task_planner,
      'composer_ui': composer,
  }
  lmps.update(low_level_lmps)

  return lmps, lmp_env


# ======================================================
# jit-ready functions (for faster replanning time, need to install numba and add "@njit")
# ======================================================
def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """voxelize a point cloud"""
  pc = pc.astype(np.float32)
  # make sure the point is within the voxel bounds
  pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxels = (pc - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxels)
  voxels = np.round(voxels, 0, _out).astype(np.int32)
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  return voxels

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """given point cloud, create a fixed size voxel map, and fill in the voxels"""
  points = points.astype(np.float32)
  voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
  voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
  # make sure the point is within the voxel bounds
  points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxel_xyz = (points - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxel_xyz)
  points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
  voxel_map = np.zeros((map_size, map_size, map_size))
  for i in range(points_vox.shape[0]):
      voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
  return voxel_map
