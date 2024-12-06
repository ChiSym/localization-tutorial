import jax.numpy as jnp
import jax
import pytest
from robot_2.where_am_i import (
    World,
    Pose,
    RobotCapabilities,
    execute_control,
    walls_to_jax,
)
import robot_2.where_am_i as where_am_i
from jax.random import PRNGKey


def test_basic_motion():
    """Test that robot moves as expected without noise"""
    # Create walls in UI format first
    walls_list = [
        [0.0, 0.0, 0],  # Bottom wall
        [1.0, 0.0, 0],
        [1.0, 0.0, 1],  # Right wall
        [1.0, 1.0, 1],
        [1.0, 1.0, 2],  # Top wall
        [0.0, 1.0, 2],
        [0.0, 1.0, 3],  # Left wall
        [0.0, 0.0, 3],
    ]

    # Convert to JAX format
    walls, wall_vecs = walls_to_jax(walls_list)
    world = World(walls, wall_vecs)

    robot = RobotCapabilities(
        p_noise=jnp.array(0.0),
        hd_noise=jnp.array(0.0),
        sensor_noise=jnp.array(0.0),
        n_sensors=jnp.array(8),
        sensor_range=jnp.array(10.0),
    )

    start_pose = Pose(jnp.array([0.5, 0.5]), jnp.array(0.0))
    key = PRNGKey(0)
    exec_sim = jax.jit(execute_control.simulate)
    # Move forward 1 unit
    result = exec_sim(key, (world, robot, start_pose, jnp.array([1.0, 0.0])))
    new_pose = result.get_retval()[0]

    assert new_pose.p[0] == pytest.approx(
        1.0 - where_am_i.WALL_COLLISION_THRESHOLD
    )  # Started at 0.5, blocked by wall at 1.0
    assert new_pose.p[1] == pytest.approx(0.5)  # Y shouldn't change

    # Rotate 90 degrees (π/2 radians)
    result = exec_sim(key, (world, robot, new_pose, jnp.array([0.0, jnp.pi / 2])))

    new_pose = result.get_retval()[0]
    assert new_pose.hd == pytest.approx(jnp.pi / 2)


def test_pose_methods():
    """Test Pose step_along and rotate methods"""
    p = Pose(jnp.array([1.0, 1.0]), jnp.array(0.0))

    # Step along heading 0 (right)
    p2 = p.step_along(jnp.array(1.0))
    assert p2.p[0] == pytest.approx(2.0)
    assert p2.p[1] == pytest.approx(1.0)

    # Rotate 90 degrees and step
    p3 = p.rotate(jnp.array(jnp.pi / 2)).step_along(jnp.array(1.0))
    assert p3.p[0] == pytest.approx(1.0)
    assert p3.p[1] == pytest.approx(2.0)


def test_walls_to_jax():
    """Test wall conversion from UI format to JAX format"""
    walls_list = [[0.0, 0.0, 0], [1.0, 0.0, 0], [1.0, 0.0, 1], [1.0, 1.0, 1]]

    walls, wall_vecs = walls_to_jax(walls_list)

    # Check shapes
    assert walls.shape == (3, 2, 2)  # 3 segments, 2 points per segment, 2 coordinates
    assert wall_vecs.shape == (3, 2)  # 3 segments, 2 coordinates per vector

    # Check first wall segment
    assert jnp.allclose(walls[0, 0], jnp.array([0.0, 0.0]))
    assert jnp.allclose(walls[0, 1], jnp.array([1.0, 0.0]))

    # Check wall vector
    assert jnp.allclose(wall_vecs[0], jnp.array([1.0, 0.0]))


if __name__ == "__main__":
    pytest.main(["-v"])
