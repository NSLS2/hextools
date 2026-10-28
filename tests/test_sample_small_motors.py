"""Verify the collection's small-sample motors without running the live profile."""

import ast
from pathlib import Path

import pytest
from ophyd_async.core import set_mock_value

import hextools
from hextools.gui.device_sources import NamespaceDeviceSource
from hextools.motors import AsyncEpicsMotor


@pytest.mark.parametrize("axis", ["x", "y"])
async def test_sample_small_motor_profile_binding_and_mock_move(axis):
    profile = Path(hextools.__file__).parent / "profiles" / "collection.py"
    variable = f"sample_small_{axis}"
    calls = [
        node.value
        for node in ast.walk(ast.parse(profile.read_text()))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
    ]
    assert len(calls) == 1, f"{variable} must be constructed in the collection"
    call = calls[0]
    assert isinstance(call.func, ast.Name) and call.func.id == "AsyncEpicsMotor"
    assert all(kw.arg is not None for kw in call.keywords)
    kwargs = {
        kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg is not None
    }
    motor = AsyncEpicsMotor(
        *(ast.literal_eval(arg) for arg in call.args),
        **kwargs,
    )
    prefix = f"XF:27IDF-OP:1{{MC:3-Ax:SampleS{axis.upper()}}}Mtr"
    assert motor.user_readback.source == f"ca://{prefix}.RBV"
    assert motor.user_setpoint.source == f"ca://{prefix}.VAL"
    assert motor.motor_stop.source == f"ca://{prefix}.STOP"
    assert motor.name == variable.replace("_", "-")

    source = NamespaceDeviceSource({variable: motor})
    assert variable in source.list_devices((AsyncEpicsMotor,))
    assert source.is_valid_device(variable, (AsyncEpicsMotor,))

    await motor.connect(mock=True)
    set_mock_value(motor.velocity, 100)
    set_mock_value(motor.acceleration_time, 0)
    await motor.set(1.0)
    assert await motor.user_readback.get_value() == pytest.approx(1.0)
