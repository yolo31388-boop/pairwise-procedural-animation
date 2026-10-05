"""
游戏内程序化动画生成与动作系统 - 核心模块

功能：
- generate_animation(): 基于关键帧 + 插值的程序化动画生成
- motion_synthesis():   多动作层叠加合成，带淡入淡出过渡
- kinematics():         正向 / 逆向运动学（平面关节链）
- animation_blending(): 带权重与时间的动画混合（平滑切换）
- procedural_walk():    带步态周期与地形自适应的程序化行走
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import math

Number = Union[int, float]
Value = Union[Number, Tuple[Number, ...]]


@dataclass
class Config:
    pass


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------

def _is_seq(value) -> bool:
    return isinstance(value, (list, tuple))


def _lerp_value(a: Value, b: Value, t: float) -> Value:
    """对标量或等长向量做线性插值。"""
    if _is_seq(a) and _is_seq(b):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    return a + (b - a) * t


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _normalize_keyframes(keyframes) -> List[Dict]:
    """接受 [(t, v), ...] 或 [{"time": t, "value": v}, ...]，按时间排序。"""
    normalized = []
    for kf in keyframes:
        if isinstance(kf, dict):
            normalized.append({"time": float(kf["time"]), "value": kf["value"]})
        else:
            t, v = kf
            normalized.append({"time": float(t), "value": v})
    normalized.sort(key=lambda kf: kf["time"])
    return normalized


# ---------------------------------------------------------------------------
# 主系统
# ---------------------------------------------------------------------------

class ProceduralAnimation:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {"time": 0.0, "blends": []}
        self._history = []

    # -- 生命周期 ---------------------------------------------------------

    def update(self, dt: float):
        """推进系统时间，并更新所有进行中的时间混合。"""
        dt = max(0.0, float(dt))
        self._state["time"] += dt
        for blend in self._state["blends"]:
            if blend["type"] == "timed":
                blend["elapsed"] = min(blend["elapsed"] + dt, blend["duration"])

    def reset(self):
        self._state = {"time": 0.0, "blends": []}
        self._history = []

    # -- 动画生成：关键帧 + 插值 ------------------------------------------

    def generate_animation(self, name: str, keyframes,
                           interpolation: str = "linear") -> Dict:
        """由关键帧生成动画片段。

        keyframes: [(time, value), ...] 或 [{"time":..., "value":...}, ...]
        interpolation: "linear" 或 "smooth"（smoothstep 平滑过渡）
        """
        if interpolation not in ("linear", "smooth"):
            raise ValueError(f"未知插值方式: {interpolation}")
        kfs = _normalize_keyframes(keyframes)
        if len(kfs) < 2:
            raise ValueError("至少需要两个关键帧才能生成动画")
        clip = {
            "name": name,
            "keyframes": kfs,
            "duration": kfs[-1]["time"],
            "interpolation": interpolation,
        }
        self._history.append(("generate_animation", name))
        return clip

    def sample_animation(self, clip: Dict, t: float) -> Value:
        """在任意时刻采样动画，关键帧之间按插值方式平滑过渡。"""
        kfs = clip["keyframes"]
        t = float(t)
        if t <= kfs[0]["time"]:
            return kfs[0]["value"]
        if t >= kfs[-1]["time"]:
            return kfs[-1]["value"]
        for i in range(len(kfs) - 1):
            a, b = kfs[i], kfs[i + 1]
            if a["time"] <= t <= b["time"]:
                span = b["time"] - a["time"]
                local = 0.0 if span <= 0 else (t - a["time"]) / span
                if clip.get("interpolation") == "smooth":
                    local = _smoothstep(local)
                return _lerp_value(a["value"], b["value"], local)
        return kfs[-1]["value"]

    # -- 动作合成：层 + 过渡 ----------------------------------------------

    def motion_synthesis(self, layers: Sequence[Dict]) -> Dict:
        """把多个动作层叠加合成为一个动作。

        每层支持:
          clip:     generate_animation 生成的片段（必填）
          weight:   层权重，默认 1.0
          start:    层开始时间，默认 0
          fade_in:  淡入时长（过渡），默认 0
          fade_out: 淡出时长（过渡），默认 0
        """
        if not layers:
            raise ValueError("至少需要一个动作层")
        normalized = []
        duration = 0.0
        for layer in layers:
            clip = layer["clip"]
            entry = {
                "clip": clip,
                "weight": float(layer.get("weight", 1.0)),
                "start": float(layer.get("start", 0.0)),
                "fade_in": max(0.0, float(layer.get("fade_in", 0.0))),
                "fade_out": max(0.0, float(layer.get("fade_out", 0.0))),
            }
            normalized.append(entry)
            duration = max(duration, entry["start"] + clip["duration"])
        motion = {"layers": normalized, "duration": duration}
        self._history.append(("motion_synthesis", len(normalized)))
        return motion

    @staticmethod
    def _layer_fade(layer: Dict, t: float) -> float:
        """层在时刻 t 的过渡系数（0~1），实现淡入淡出。"""
        local = t - layer["start"]
        if local < 0.0:
            return 0.0
        end = layer["clip"]["duration"]
        factor = 1.0
        if layer["fade_in"] > 0.0 and local < layer["fade_in"]:
            factor = min(factor, local / layer["fade_in"])
        if layer["fade_out"] > 0.0:
            remaining = end - local
            if remaining < layer["fade_out"]:
                factor = min(factor, max(0.0, remaining / layer["fade_out"]))
        return factor

    def sample_motion(self, motion: Dict, t: float) -> Value:
        """采样合成动作：各层按权重与过渡系数叠加。"""
        t = float(t)
        result = 0.0
        for layer in motion["layers"]:
            local = t - layer["start"]
            value = self.sample_animation(layer["clip"], local)
            scale = layer["weight"] * self._layer_fade(layer, t)
            if _is_seq(value):
                scaled = tuple(v * scale for v in value)
                if not _is_seq(result):
                    result = tuple(0.0 for _ in value)
                result = tuple(r + s for r, s in zip(result, scaled))
            else:
                result = result + value * scale
        return result

    # -- 运动学：正向 + 逆向 ----------------------------------------------

    def kinematics(self, mode: str, **kwargs):
        """运动学入口。

        mode="forward"/"fk": 需 angles, lengths，由关节角度计算末端位置
        mode="inverse"/"ik": 需 target, lengths，由末端位置计算关节角度
        """
        if mode in ("forward", "fk"):
            return self.forward_kinematics(kwargs["angles"], kwargs["lengths"])
        if mode in ("inverse", "ik"):
            return self.inverse_kinematics(kwargs["target"], kwargs["lengths"])
        raise ValueError(f"未知运动学模式: {mode}")

    def forward_kinematics(self, angles: Sequence[Number],
                           lengths: Sequence[Number]) -> Dict:
        """正向运动学：平面关节链，角度为相对父关节的弧度。"""
        if len(angles) != len(lengths):
            raise ValueError("关节角度与连杆长度数量必须一致")
        positions = [(0.0, 0.0)]
        x = y = 0.0
        absolute = 0.0
        for angle, length in zip(angles, lengths):
            absolute += float(angle)
            x += float(length) * math.cos(absolute)
            y += float(length) * math.sin(absolute)
            positions.append((x, y))
        return {"positions": positions, "end_effector": (x, y)}

    def inverse_kinematics(self, target: Tuple[Number, Number],
                           lengths: Sequence[Number],
                           tol: float = 1e-6) -> Optional[List[float]]:
        """逆向运动学：由末端目标位置求关节角度。

        两连杆使用解析解；更多连杆使用 CCD 迭代。目标不可达返回 None。
        """
        lengths = [float(l) for l in lengths]
        tx, ty = float(target[0]), float(target[1])
        dist = math.hypot(tx, ty)
        total = sum(lengths)
        min_reach = abs(lengths[0] - sum(lengths[1:])) if lengths else 0.0
        if dist > total + tol or dist < min_reach - tol:
            return None
        if len(lengths) == 2:
            return self._ik_two_link(tx, ty, lengths[0], lengths[1])
        return self._ik_ccd(tx, ty, lengths, tol)

    @staticmethod
    def _ik_two_link(tx: float, ty: float, l1: float, l2: float) -> List[float]:
        d2 = tx * tx + ty * ty
        cos_a2 = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_a2 = max(-1.0, min(1.0, cos_a2))
        a2 = math.acos(cos_a2)
        a1 = math.atan2(ty, tx) - math.atan2(l2 * math.sin(a2),
                                             l1 + l2 * math.cos(a2))
        return [a1, a2]

    def _ik_ccd(self, tx: float, ty: float, lengths: List[float],
                tol: float, max_iter: int = 200) -> Optional[List[float]]:
        angles = [0.0] * len(lengths)
        for _ in range(max_iter):
            fk = self.forward_kinematics(angles, lengths)
            ex, ey = fk["end_effector"]
            if math.hypot(tx - ex, ty - ey) <= tol:
                return angles
            joints = fk["positions"]
            for i in range(len(angles) - 1, -1, -1):
                jx, jy = joints[i]
                to_end = math.atan2(ey - jy, ex - jx)
                to_target = math.atan2(ty - jy, tx - jx)
                angles[i] += to_target - to_end
                fk = self.forward_kinematics(angles, lengths)
                ex, ey = fk["end_effector"]
        fk = self.forward_kinematics(angles, lengths)
        ex, ey = fk["end_effector"]
        return angles if math.hypot(tx - ex, ty - ey) <= 1e-3 else None

    # -- 动画混合：权重 + 时间 --------------------------------------------

    def animation_blending(self, anim_from: Dict, anim_to: Dict,
                           weight: Optional[float] = None,
                           blend_time: Optional[float] = None) -> Dict:
        """创建动画混合。

        给定 weight 时为静态加权混合（0=from，1=to）；
        给定 blend_time 时为时间混合，权重随 update(dt) 从 0 平滑过渡到 1。
        """
        if weight is None and blend_time is None:
            raise ValueError("必须指定 weight 或 blend_time")
        if weight is not None:
            blend = {"type": "static", "from": anim_from, "to": anim_to,
                     "weight": max(0.0, min(1.0, float(weight)))}
        else:
            blend = {"type": "timed", "from": anim_from, "to": anim_to,
                     "elapsed": 0.0, "duration": max(1e-9, float(blend_time))}
            self._state["blends"].append(blend)
        self._history.append(("animation_blending", blend["type"]))
        return blend

    def blend_weight(self, blend: Dict) -> float:
        """当前混合权重（时间混合使用 smoothstep 平滑过渡）。"""
        if blend["type"] == "static":
            return blend["weight"]
        return _smoothstep(blend["elapsed"] / blend["duration"])

    def sample_blend(self, blend: Dict, t: float) -> Value:
        """按当前权重采样混合结果。"""
        w = self.blend_weight(blend)
        a = self.sample_animation(blend["from"], t)
        b = self.sample_animation(blend["to"], t)
        return _lerp_value(a, b, w)

    # -- 程序化行走：步态 + 地形自适应 ------------------------------------

    def procedural_walk(self, speed: Number, time: Number = 0.0,
                        terrain=None) -> Dict:
        """程序化行走。

        speed:   移动速度（决定步频与步幅）
        time:    当前时间（决定步态相位）
        terrain: 地形，可为高度函数 h(x)、{"heights": [...], "spacing": s}
                 或 {"slope": m, "offset": c}
        返回步态相位、步频、步幅以及左右脚位置（含地形自适应）。
        """
        speed = max(0.0, float(speed))
        time = float(time)
        base_stride = float(self.config.get("stride_length", 1.0))
        base_step_h = float(self.config.get("step_height", 0.2))

        stride = base_stride * min(1.5, 0.5 + 0.5 * math.sqrt(speed)) \
            if speed > 0 else base_stride
        frequency = speed / stride if speed > 0 else 0.0
        phase = (time * frequency) % 1.0

        body_x = speed * time
        slope = self._terrain_slope(terrain, body_x)
        adapt = 1.0 / (1.0 + abs(slope))
        stride *= adapt
        step_height = base_step_h * (1.0 + abs(slope))
        if speed > 0:
            frequency = speed / stride
            phase = (time * frequency) % 1.0

        feet = {}
        for label, offset in (("left_foot", 0.0), ("right_foot", 0.5)):
            foot_phase = (phase + offset) % 1.0
            angle = 2.0 * math.pi * foot_phase
            foot_x = body_x + 0.5 * stride * math.cos(angle)
            lift = step_height * max(0.0, math.sin(angle))
            ground = self._terrain_height(terrain, foot_x)
            feet[label] = (foot_x, ground + lift)

        return {
            "phase": phase,
            "frequency": frequency,
            "stride_length": stride,
            "step_height": step_height,
            "slope": slope,
            "terrain_adapted": terrain is not None,
            **feet,
        }

    @staticmethod
    def _terrain_height(terrain, x: float) -> float:
        if terrain is None:
            return 0.0
        if callable(terrain):
            return float(terrain(x))
        if isinstance(terrain, dict):
            if "heights" in terrain:
                heights = terrain["heights"]
                spacing = float(terrain.get("spacing", 1.0))
                pos = max(0.0, x / spacing)
                i = int(pos)
                if i >= len(heights) - 1:
                    return float(heights[-1])
                frac = pos - i
                return float(heights[i]) * (1 - frac) + float(heights[i + 1]) * frac
            if "slope" in terrain:
                return float(terrain["slope"]) * x + float(terrain.get("offset", 0.0))
        return 0.0

    def _terrain_slope(self, terrain, x: float, dx: float = 0.1) -> float:
        if terrain is None:
            return 0.0
        h1 = self._terrain_height(terrain, x - dx)
        h2 = self._terrain_height(terrain, x + dx)
        return (h2 - h1) / (2.0 * dx)
