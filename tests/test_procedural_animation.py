"""
游戏内程序化动画生成与动作系统 - 验收测试
运行方式：python -m pytest tests/test_procedural_animation.py -q
共 16 个测试用例
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animation.procedural import ProceduralAnimation


class TestProceduralAnimation:
    def setup_method(self):
        self.system = ProceduralAnimation()

    # -- 动画生成：关键帧 + 插值 ------------------------------------------

    def test_case_01(self):
        # 生成动画包含关键帧与时长，不再是预录数据
        clip = self.system.generate_animation(
            "walk", [(0.0, 0.0), (1.0, 10.0), (2.0, 0.0)])
        assert clip["name"] == "walk"
        assert len(clip["keyframes"]) == 3
        assert clip["duration"] == pytest.approx(2.0)

    def test_case_02(self):
        # 线性插值：关键帧之间有平滑过渡
        clip = self.system.generate_animation(
            "move", [(0.0, 0.0), (2.0, 10.0)], interpolation="linear")
        assert self.system.sample_animation(clip, 1.0) == pytest.approx(5.0)
        assert self.system.sample_animation(clip, 0.5) == pytest.approx(2.5)

    def test_case_03(self):
        # 平滑插值：端点精确，中点仍为一半（smoothstep 对称）
        clip = self.system.generate_animation(
            "ease", [(0.0, 0.0), (1.0, 4.0)], interpolation="smooth")
        assert self.system.sample_animation(clip, 0.0) == pytest.approx(0.0)
        assert self.system.sample_animation(clip, 1.0) == pytest.approx(4.0)
        assert self.system.sample_animation(clip, 0.5) == pytest.approx(2.0)
        # 平滑插值在起点附近变化更缓（导数为 0）
        assert self.system.sample_animation(clip, 0.1) < 4.0 * 0.1

    def test_case_04(self):
        # 采样超出关键帧范围时钳位到端点；支持向量值
        clip = self.system.generate_animation(
            "vec", [(0.0, (0.0, 0.0)), (1.0, (2.0, 4.0))])
        assert self.system.sample_animation(clip, -1.0) == (0.0, 0.0)
        assert self.system.sample_animation(clip, 5.0) == (2.0, 4.0)
        mid = self.system.sample_animation(clip, 0.5)
        assert mid == pytest.approx((1.0, 2.0))

    # -- 动作合成：层 + 过渡 ----------------------------------------------

    def test_case_05(self):
        # 多个动作层按权重叠加混合
        walk = self.system.generate_animation("walk", [(0.0, 1.0), (1.0, 1.0)])
        wave = self.system.generate_animation("wave", [(0.0, 10.0), (1.0, 10.0)])
        motion = self.system.motion_synthesis([
            {"clip": walk, "weight": 1.0},
            {"clip": wave, "weight": 0.5},
        ])
        assert len(motion["layers"]) == 2
        assert self.system.sample_motion(motion, 0.5) == pytest.approx(1.0 + 5.0)

    def test_case_06(self):
        # 层过渡：fade_in 期间权重从 0 平滑爬升，不是瞬间出现
        base = self.system.generate_animation("base", [(0.0, 0.0), (2.0, 0.0)])
        top = self.system.generate_animation("top", [(0.0, 8.0), (2.0, 8.0)])
        motion = self.system.motion_synthesis([
            {"clip": base},
            {"clip": top, "fade_in": 1.0},
        ])
        assert self.system.sample_motion(motion, 0.0) == pytest.approx(0.0)
        assert self.system.sample_motion(motion, 0.5) == pytest.approx(4.0)
        assert self.system.sample_motion(motion, 1.5) == pytest.approx(8.0)

    def test_case_07(self):
        # 合成时长覆盖所有层（含 start 偏移）；fade_out 平滑收尾
        a = self.system.generate_animation("a", [(0.0, 2.0), (1.0, 2.0)])
        b = self.system.generate_animation("b", [(0.0, 4.0), (2.0, 4.0)])
        motion = self.system.motion_synthesis([
            {"clip": a, "fade_out": 1.0},
            {"clip": b, "start": 1.0},
        ])
        assert motion["duration"] == pytest.approx(3.0)
        # t=0.5：a 淡出到一半权重
        assert self.system.sample_motion(motion, 0.5) == pytest.approx(1.0)
        # t=2.0：a 已结束，b 全量
        assert self.system.sample_motion(motion, 2.0) == pytest.approx(4.0)

    # -- 运动学：正向 + 逆向 ----------------------------------------------

    def test_case_08(self):
        # 正向运动学：根据关节角度计算末端位置
        fk = self.system.kinematics(
            "forward", angles=[0.0, 0.0], lengths=[1.0, 1.0])
        assert fk["end_effector"] == pytest.approx((2.0, 0.0))
        fk = self.system.kinematics(
            "forward", angles=[math.pi / 2, 0.0], lengths=[1.0, 1.0])
        assert fk["end_effector"] == pytest.approx((0.0, 2.0), abs=1e-9)

    def test_case_09(self):
        # 逆向运动学：根据末端位置反解关节角度，并与正解互相验证
        lengths = [1.0, 1.0]
        target = (1.0, 1.0)
        angles = self.system.kinematics("inverse", target=target, lengths=lengths)
        assert angles is not None
        fk = self.system.kinematics("forward", angles=angles, lengths=lengths)
        assert fk["end_effector"] == pytest.approx(target, abs=1e-6)

    def test_case_10(self):
        # 目标不可达时返回 None；多连杆链用 CCD 求解
        assert self.system.kinematics(
            "inverse", target=(5.0, 0.0), lengths=[1.0, 1.0]) is None
        angles = self.system.kinematics(
            "inverse", target=(1.5, 0.5), lengths=[1.0, 0.8, 0.6])
        assert angles is not None
        fk = self.system.kinematics(
            "forward", angles=angles, lengths=[1.0, 0.8, 0.6])
        assert fk["end_effector"] == pytest.approx((1.5, 0.5), abs=1e-3)

    # -- 动画混合：权重 + 时间 --------------------------------------------

    def test_case_11(self):
        # 静态权重混合：0=动画A，1=动画B，0.5=两者中点
        idle = self.system.generate_animation("idle", [(0.0, 0.0), (1.0, 0.0)])
        run = self.system.generate_animation("run", [(0.0, 10.0), (1.0, 10.0)])
        b0 = self.system.animation_blending(idle, run, weight=0.0)
        b1 = self.system.animation_blending(idle, run, weight=1.0)
        b5 = self.system.animation_blending(idle, run, weight=0.5)
        assert self.system.sample_blend(b0, 0.5) == pytest.approx(0.0)
        assert self.system.sample_blend(b1, 0.5) == pytest.approx(10.0)
        assert self.system.sample_blend(b5, 0.5) == pytest.approx(5.0)

    def test_case_12(self):
        # 时间混合：切换不是瞬间的，权重随 update(dt) 渐进变化
        idle = self.system.generate_animation("idle", [(0.0, 0.0), (1.0, 0.0)])
        run = self.system.generate_animation("run", [(0.0, 10.0), (1.0, 10.0)])
        blend = self.system.animation_blending(idle, run, blend_time=1.0)
        assert self.system.blend_weight(blend) == pytest.approx(0.0)
        self.system.update(0.5)
        w = self.system.blend_weight(blend)
        assert 0.0 < w < 1.0
        assert w == pytest.approx(0.5)
        assert self.system.sample_blend(blend, 0.5) == pytest.approx(5.0)

    def test_case_13(self):
        # 混合时间结束后权重到达 1，完全切换到目标动画
        idle = self.system.generate_animation("idle", [(0.0, 0.0), (1.0, 0.0)])
        run = self.system.generate_animation("run", [(0.0, 10.0), (1.0, 10.0)])
        blend = self.system.animation_blending(idle, run, blend_time=0.5)
        self.system.update(0.25)
        self.system.update(0.25)
        self.system.update(1.0)  # 超时不应超过 1
        assert self.system.blend_weight(blend) == pytest.approx(1.0)
        assert self.system.sample_blend(blend, 0.5) == pytest.approx(10.0)

    # -- 程序化行走：步态 + 自适应 ----------------------------------------

    def test_case_14(self):
        # 步态周期：相位随时间推进且落在 [0,1)，步频随速度提升
        slow = self.system.procedural_walk(speed=1.0, time=0.3)
        fast = self.system.procedural_walk(speed=3.0, time=0.3)
        assert 0.0 <= slow["phase"] < 1.0
        assert fast["frequency"] > slow["frequency"]
        later = self.system.procedural_walk(speed=1.0, time=0.8)
        assert later["phase"] != pytest.approx(slow["phase"])

    def test_case_15(self):
        # 步态交替：左右脚反相，一脚抬起时另一脚落地（不再像机器人平移）
        gait = None
        for i in range(1, 100):
            g = self.system.procedural_walk(speed=1.0, time=i * 0.01)
            if g["left_foot"][1] > 0.05:
                gait = g
                break
        assert gait is not None
        assert gait["left_foot"][1] > gait["right_foot"][1]
        assert gait["right_foot"][1] == pytest.approx(0.0, abs=1e-9)

    def test_case_16(self):
        # 地形自适应：脚贴合地形高度，上坡时步幅缩短、抬腿更高
        flat = self.system.procedural_walk(speed=1.0, time=0.3)
        slope = self.system.procedural_walk(
            speed=1.0, time=0.3, terrain={"slope": 0.5})
        assert slope["terrain_adapted"] is True
        assert slope["stride_length"] < flat["stride_length"]
        assert slope["step_height"] > flat["step_height"]
        # 脚掌不低于落点处的地面高度
        for foot in ("left_foot", "right_foot"):
            x, y = slope[foot]
            assert y >= 0.5 * x - 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
