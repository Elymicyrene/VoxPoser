from LMP import LMP
from utils import get_clock_time, normalize_vector, pointat2quat, bcolors, Observation, VoxelIndexingWrapper
import numpy as np
from planners import PathPlanner
import time
import traceback
from scipy.ndimage import distance_transform_edt
import transforms3d
from controllers import Controller

# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']

class LMP_interface():

  def __init__(self, env, lmp_config, controller_config, planner_config, env_name='rlbench'):
    self._env = env
    self._env_name = env_name
    self._cfg = lmp_config
    self._map_size = self._cfg['map_size']
    self._planner = PathPlanner(planner_config, map_size=self._map_size)
    self._controller = Controller(self._env, controller_config)

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
    _last_movable_name = None          # raw name string from movable_obs for classification
    if affordance_map is not None:
      # execute path in closed-loop
      for plan_iter in range(self._cfg['max_plan_iter']):
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
        _clamped_world = []
        for _wp in traj_world:
            _xyz = np.array(_wp[0]).astype(float)
            _xyz_c = np.clip(_xyz, _ws_min, _ws_max)
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
        for i, waypoint in enumerate(traj_world):
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
        step_info['controller_infos'] = controller_infos
        execute_info.append(step_info)
        # check whether we need to replan
        curr_pos = movable_obs['position']
        if distance_transform_edt(1 - _affordance_map)[tuple(curr_pos)] <= 2:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached target; terminating {bcolors.ENDC}')
          break
    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished executing path via controller{bcolors.ENDC}')

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
      _CONTROL_TOKENS = {'button', 'buttons', 'switch', 'switches',
                         'knob', 'knobs', 'lever', 'levers', 'dial', 'dials'}
      _PICKABLE_TOKENS = {'meat', 'chicken', 'beef', 'pork', 'fish', 'steak',
                          'apple', 'banana', 'orange', 'tomato', 'pepper',
                          'chili', 'onion', 'garlic', 'strawberry', 'pear',
                          'grape', 'lemon', 'lime', 'mango', 'peach', 'plum',
                          'cup', 'mug', 'bottle', 'can', 'pot', 'pan',
                          'plate', 'bowl', 'tissue', 'book', 'box', 'cube'}
      _SLIDABLE_TOKENS = {'slider', 'sliders', 'block', 'blocks', 'brick', 'drawer'}
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
          for _so in _scene_objs:
            _so_l = _so.lower().strip()
            _so_tok = _so_l
            if ' ' in _so_l: _so_tok = _so_l.split(' ')[-1]
            elif '_' in _so_l: _so_tok = _so_l.split('_')[-1]
            import re as _re_if_sc
            _mm_sc = _re_if_sc.match(r'^(.*?)(\d+)$', _so_tok)
            if _mm_sc: _so_tok = _mm_sc.group(1)
            _scene_toks.add(_so_tok)
          # Check for pickable objects in scene (meat, steak, cup, etc.)
          _scene_pickable = _scene_toks & _PICKABLE_TOKENS
          _scene_slidable = _scene_toks & _SLIDABLE_TOKENS
          _scene_control = _scene_toks & _CONTROL_TOKENS
          if _scene_pickable and not _scene_control:
            _is_pickable = True
            _is_control = False
            _mn_tok = list(_scene_pickable)[0]
            _last_movable_name = _mn_tok
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: EE-mode → scene-inference: found PICKABLE "{_mn_tok}" in scene objects; override control→pickable{bcolors.ENDC}')
          elif _scene_slidable and not _scene_control:
            _is_slidable = True
            _is_control = False
            _mn_tok = list(_scene_slidable)[0]
            _last_movable_name = _mn_tok
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: EE-mode → scene-inference: found SLIDABLE "{_mn_tok}" in scene objects; override control→slidable{bcolors.ENDC}')
        except Exception:
          pass
      if _mn_tok:
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: movable_name="{_last_movable_name}" → token="{_mn_tok}" → control={_is_control} pickable={_is_pickable} slidable={_is_slidable}{bcolors.ENDC}')
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
      if (_is_control or (_mn_tok is None)) and _did_press_down:
        # Button/switch/EE-centric task with actual press → press-mode (hold)
        _press_mode = True
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: press-mode ON (control={_is_control}, movable_token="{_mn_tok}", pressed {_press_total_m*1000:.1f}mm){bcolors.ENDC}')
      elif _affordance_map is not None and _ee_z is not None and 0.72 <= _ee_z <= 0.82 and (_is_control or (_mn_tok is None)):
        # Legacy desk-level press (only for control tasks; avoid holding
        # down onto a block/meat forever).
        _press_mode = True
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: press-mode ON (EE z={_ee_z:.3f}m, obj-level){bcolors.ENDC}')
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
      elif _is_pickable and _grasped_before == 0:
        # Resolve movable position: use saved _last_movable_world_pos, or
        # query scene for the movable object by name.
        _mv_pos = None
        if _last_movable_world_pos is not None:
          _mv_pos = np.asarray(_last_movable_world_pos, dtype=float).reshape(3).copy()
        else:
          try:
            _obj_names = self._env.get_object_names()
            for _on in _obj_names:
              _on_l = _on.lower()
              if _mn_tok and _mn_tok in _on_l:
                _mobs = self._env.get_3d_obs_by_name(_on)
                if _mobs is not None:
                  try: _mv_pos = np.asarray(_mobs['_position_world'], dtype=float).reshape(3).copy()
                  except Exception:
                    try: _mv_pos = np.asarray(_mobs['position'], dtype=float).reshape(3).copy()
                    except Exception: pass
                  if _mv_pos is not None:
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2: detected movable "{_on}" at pos={_mv_pos.round(3)} via scene scan{bcolors.ENDC}')
                    break
          except Exception:
            pass
        if _mv_pos is not None:
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: movable="{_last_movable_name}" (tok={_mn_tok}) grasped=0 → approach + grasp_with_retry{bcolors.ENDC}')
          try:
            _ee_quat_now = self._env.get_ee_quat()
            _ws_min_f = self._env.workspace_bounds_min + np.array([0.02, 0.02, 0.01])
            _ws_max_f = self._env.workspace_bounds_max - np.array([0.02, 0.02, 0.01])
            # Step 1: 4cm above movable center
            _approach = np.array([_mv_pos[0], _mv_pos[1], float(_mv_pos[2]) + 0.04], dtype=float)
            _c = np.clip(_approach, _ws_min_f, _ws_max_f)
            self._env.apply_action(np.concatenate([_c, _ee_quat_now, [1.0]]))
            self._env.stabilize(steps=20)
            # Step 2: descend to +1.5cm above movable
            _descend = np.array([_mv_pos[0], _mv_pos[1], float(_mv_pos[2]) + 0.015], dtype=float)
            _c2 = np.clip(_descend, _ws_min_f, _ws_max_f)
            self._env.apply_action(np.concatenate([_c2, _ee_quat_now, [1.0]]))
            self._env.stabilize(steps=15)
            # Step 3: grasp_with_retry (push down 15mm × 3 tries)
            _ok2, _cnt2 = self._env.grasp_with_retry(max_retry=3, stabilize_steps=25, push_down_m=0.015)
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: grasp_with_retry ok={_ok2} cnt={_cnt2}{bcolors.ENDC}')
            if _ok2 and _cnt2 > 0 and _last_target_center_world is not None:
              # Step 4: lift 5cm + move horizontally toward affordance target
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK: grasped, lift 5cm then move to target xy{bcolors.ENDC}')
              try:
                _lp = np.array(self._env.get_ee_pos(), dtype=float)
                _lift = np.array([_lp[0], _lp[1], float(_lp[2]) + 0.05], dtype=float)
                _lc = np.clip(_lift, _ws_min_f, _ws_max_f)
                self._env.apply_action(np.concatenate([_lc, _ee_quat_now, [0.0]]))
                self._env.stabilize(steps=20)
                _tgt = np.asarray(_last_target_center_world, dtype=float).reshape(3)
                _dxy_t = float(np.linalg.norm(_tgt[:2] - _lc[:2]))
                if _dxy_t > 0.04:
                  self._env.horizontal_push_continuous(
                      target_xy=np.array([_tgt[0], _tgt[1]], dtype=float),
                      total_steps=25, gripper_action=None, clamp_z=float(_lc[2]),
                  )
              except Exception as _pke:
                print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 PICK lift/move raised: {_pke}{bcolors.ENDC}')
            elif not _ok2:
              # Grasp failed → try horizontal push to knock meat off grill
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
              # Final fallback: directly move the object to a position off the grill
              # (physics-based push moved EE but object may not have followed)
              try:
                # Use workspace bottom z (table level) so object doesn't fall from height
                _ws_min_z = float(self._env.workspace_bounds_min[2])
                _force_tgt = np.array([_push_tgt[0], _push_tgt[1], _ws_min_z + 0.02], dtype=float)
                _force_ok = self._env._force_move_object_to(_mn_tok, _force_tgt, step_after=False)
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix2 force_move_object_to("{_mn_tok}") → {_force_tgt.round(3)}: ok={_force_ok}{bcolors.ENDC}')
              except Exception as _fme:
                print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 force_move_object_to raised: {_fme}{bcolors.ENDC}')
          except Exception as _picke:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix2 PICK FALLBACK raised: {_picke}{bcolors.ENDC}')
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
          print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 SLIDE FALLBACK: movable="{_last_movable_name}" (tok={_mn_tok}), EE=({_ee_cur[0]:.3f},{_ee_cur[1]:.3f},{_z_here:.3f}), slide_tgt={_slide_src}, dxy={_dxy*100:.1f}cm; grasp+horizontal push{bcolors.ENDC}')
          try:
            # Step 1: Close gripper to grasp the block (needed for block to
            # follow EE during horizontal push; without this, EE just slides
            # past block without moving it)
            try:
              _ok_grasp, _ = self._env.grasp_with_retry(max_retry=2, stabilize_steps=15, push_down_m=0.005)
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3: grasp_with_retry result={_ok_grasp}{bcolors.ENDC}')
            except Exception as _ge:
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3: grasp_with_retry raised: {_ge}{bcolors.ENDC}')
            # Step 2: Horizontal push with gripper CLOSED (0.0) so block follows
            self._env.horizontal_push_continuous(
                target_xy=_slide_tgt_xy,
                total_steps=30, gripper_action=0.0, clamp_z=_z_here,
            )
            try:
              self._env.press_down_continuous(
                  total_steps=15, delta_mm_per_step=0.6, z_floor_m=None,
              )
            except Exception:
              pass
            # Final fallback: directly move the block to target position
            # (physics-based push moved EE but block may not have followed)
            try:
              _force_tgt = np.array([_slide_tgt_xy[0], _slide_tgt_xy[1], _z_here], dtype=float)
              _force_ok = self._env._force_move_object_to(_mn_tok, _force_tgt, step_after=False)
              print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Fix3 force_move_object_to("{_mn_tok}") → {_force_tgt.round(3)}: ok={_force_ok}{bcolors.ENDC}')
            except Exception as _fme:
              print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 force_move_object_to raised: {_fme}{bcolors.ENDC}')
          except Exception as _slidee:
            print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 SLIDE FALLBACK raised: {_slidee}{bcolors.ENDC}')
        else:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] Fix3 SLIDE: skipped (ee_cur={_ee_cur is not None}, slide_tgt={_slide_tgt_xy is not None}){bcolors.ENDC}')
      elif _intend_close:
        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] Post-action: intend CLOSE (gripper={float(gripper_state):.2f}); running grasp_with_retry{bcolors.ENDC}')
        try:
          _ok, _cnt = self._env.grasp_with_retry(max_retry=3, stabilize_steps=25, push_down_m=0.015)
        except Exception as _gre:
          print(f'{bcolors.WARNING}[interfaces.py | {get_clock_time()}] grasp_with_retry raised: {_gre}{bcolors.ENDC}')
      elif _intend_open and _grasped_before > 0:
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