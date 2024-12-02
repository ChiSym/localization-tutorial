import jax.numpy as jnp
import pytest
from robot_2.where_am_i import World, Pose, RobotCapabilities, execute_control
import robot_2.where_am_i as where_am_i
from jax.random import PRNGKey


def test_basic_motion():
    """Test that robot moves as expected without noise"""
    # Convert walls to JAX array at creation - now in (N,2,2) shape
    walls = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],  # bottom wall
            [[1.0, 0.0], [1.0, 1.0]],  # right wall
            [[1.0, 1.0], [0.0, 1.0]],  # top wall
            [[0.0, 1.0], [0.0, 0.0]],  # left wall
        ]
    )
    world = World(walls)
    robot = RobotCapabilities(
        p_noise=0.0, hd_noise=0.0, sensor_noise=0.0, n_sensors=8, sensor_range=10.0
    )

    start_pose = Pose(jnp.array([0.5, 0.5]), 0.0)
    key = PRNGKey(0)

    # Move forward 1 unit
    new_pose, readings, key = execute_control(
        world=world, robot=robot, current_pose=start_pose, control=(1.0, 0.0), key=key
    )
    assert new_pose.p[0] == pytest.approx(
        1.0 - where_am_i.WALL_COLLISION_THRESHOLD
    )  # Started at 0.5, blocked by wall at 1.0
    assert new_pose.p[1] == pytest.approx(0.5)  # Y shouldn't change

    # Rotate 90 degrees (π/2 radians)
    new_pose, readings, key = execute_control(
        world=world,
        robot=robot,
        current_pose=new_pose,
        control=(0.0, jnp.pi / 2),
        key=key,
    )
    assert new_pose.hd == pytest.approx(jnp.pi / 2)


def test_pose_methods():
    """Test Pose step_along and rotate methods"""
    p = Pose(jnp.array([1.0, 1.0]), 0.0)

    # Step along heading 0 (right)
    p2 = p.step_along(1.0)
    assert p2.p[0] == pytest.approx(2.0)
    assert p2.p[1] == pytest.approx(1.0)

    # Rotate 90 degrees and step
    p3 = p.rotate(jnp.pi / 2).step_along(1.0)
    assert p3.p[0] == pytest.approx(1.0)
    assert p3.p[1] == pytest.approx(2.0)


pytest.main(["-v"])  #
