from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from ophyd_async.epics.core import EpicsDevice



class Filter1(EpicsDevice):

    def __init__(self):
        super().__init__("XF:27ID1A-OP:1{Fltr:1-Ax:", name="filter1")
        self.upstream = AsyncEpicsMotor("Yu}Mtr", name="upstream")
        self.downstream = AsyncEpicsMotor("Yd}Mtr", name="downstream")


class Slits(EpicsDevice):

    def __init__(self, prefix: str, num: int, name: str = ""):
        super().__init__(f"{prefix}{{Slt:{num}-Ax:", name=name or f"slits{num}")
        self.inboard = AsyncEpicsMotor("I}Mtr", name="inboard")
        self.outboard = AsyncEpicsMotor("O}Mtr", name="outboard")
        self.bottom = AsyncEpicsMotor("B}Mtr", name="bottom")
        self.top = AsyncEpicsMotor("T}Mtr", name="top")
        self.horiz_gap = AsyncEpicsMotor("HG}Mtr", name="horiz_gap")
        self.vert_gap = AsyncEpicsMotor("VG}Mtr", name="vert_gap")
        self.horiz_center = AsyncEpicsMotor("HC}Mtr", name="horiz_center")
        self.vert_center = AsyncEpicsMotor("VC}Mtr", name="vert_center")


class DCLM(EpicsDevice):

    def __init__(self):
        super().__init__("XF:27ID1A-OP:1{Mono:DCLM-Ax:", name="dclm")
        self.xtal2_z = AsyncEpicsMotor("Z2}Mtr", name="xtal2_z")
        self.cooling_tower_zc = AsyncEpicsMotor("ZC}Mtr", name="cooling_tower_zc")
        self.system_pitch = AsyncEpicsMotor("P}Mtr", name="system_pitch")
        self.xtal1_bend_c1a = AsyncEpicsMotor("C1A}Mtr", name="xtal1_bend_c1a")
        self.xtal1_bend_c1b = AsyncEpicsMotor("C1B}Mtr", name="xtal1_bend_c1b")
        self.xtal2_bend_c2a = AsyncEpicsMotor("C2A}Mtr", name="xtal2_bend_c2a")
        self.xtal2_bend_c2b = AsyncEpicsMotor("C2B}Mtr", name="xtal2_bend_c2b")
        self.xtal2_z_readback = AsyncEpicsMotor("Z2}Mtr.RBV", name="xtal2_z_readback")
        self.xtal2_roll = AsyncEpicsMotor("C2R}Mtr", name="xtal2_roll")
        self.xtal1_vertical_trans = AsyncEpicsMotor("C1Y}Mtr", name="xtal1_vertical_trans")
        self.xtal1_pitch = AsyncEpicsMotor("C1P}Mtr", name="xtal1_pitch")
        self.xtal2_pitch = AsyncEpicsMotor("C2P}Mtr", name="xtal2_pitch")
        self.cooled_beam_stop = AsyncEpicsMotor("BS}Mtr", name="cooled_beam_stop")
        self.fluorescent_screen = AsyncEpicsMotor("FS}Mtr", name="fluorescent_screen")

class OpticsTable(EpicsDevice):

    def __init__(self):
        super().__init__("XF:27ID1A-OP:1{OPT:1-Ax:", name="optics_table")
        self.x2 = AsyncEpicsMotor("X2}Mtr", name="x2")
        self.y2 = AsyncEpicsMotor("Y2}Mtr", name="y2")
        self.rx3 = AsyncEpicsMotor("RX3}Mtr", name="rx3")
        self.ry3 = AsyncEpicsMotor("RY3}Mtr", name="ry3")
        self.x3 = AsyncEpicsMotor("X3}Mtr", name="x3")
        self.y3 = AsyncEpicsMotor("Y3}Mtr", name="y3")
        self.ry4 = AsyncEpicsMotor("RY4}Mtr", name="ry4")
        self.x4 = AsyncEpicsMotor("X4}Mtr", name="x4")


class SampleTower(EpicsDevice):

    def __init__(self):
        super().__init__("XF:27ID1A-OP:1{SMPL:1-Ax:", name="sample_tower")
        self.y = AsyncEpicsMotor("Y}Mtr", name="y")
        self.pitch = AsyncEpicsMotor("Rx}Mtr", name="pitch")
        self.roll = AsyncEpicsMotor("Rz}Mtr", name="roll")

        # Real motors that combine to give y, pitch, and roll.
        self.x1 = AsyncEpicsMotor("X1}Mtr", name="x1")
        self.x2 = AsyncEpicsMotor("X2}Mtr", name="x2")
        self.z1 = AsyncEpicsMotor("Z1}Mtr", name="z1")
        self.z2 = AsyncEpicsMotor("Z2}Mtr", name="z2")
        self.inboard_y = AsyncEpicsMotor("Y1}Mtr", name="inboard_y")
        self.outboard_y = AsyncEpicsMotor("Y2}Mtr", name="outboard_y")
        self.downstream_y = AsyncEpicsMotor("Y3}Mtr", name="downstream_y")

