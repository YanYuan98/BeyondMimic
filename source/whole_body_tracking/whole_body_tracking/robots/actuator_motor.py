from __future__ import annotations

from isaaclab.actuators.actuator_pd import DelayedPDActuator
from isaaclab.actuators.actuator_cfg import DelayedPDActuatorCfg
from isaaclab.utils.types import ArticulationActions

from isaaclab.utils import configclass
from dataclasses import MISSING
import torch

class DelayedDCMotor(DelayedPDActuator):
    """DC motor actuator model with velocity-based saturation and command delay.

    This class combines the DC motor saturation model with command delay functionality.
    It uses critical speed instead of saturation effort for parameterization.
    """

    cfg: DelayedDCMotorCfg
    """The configuration for the delayed DC motor actuator model."""

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        # Initialize the delayed PD actuator first
        super().__init__(cfg, *args, **kwargs)

        # Parse configuration - using critical speed based parameterization
        if self.cfg.critical_speed is None:
            raise ValueError("The critical_speed must be provided for the delayed DC motor actuator model.")
        if self.cfg.velocity_limit is None:
            raise ValueError("The velocity_limit must be provided for the delayed DC motor actuator model.")
        if self.cfg.effort_limit is None:
            raise ValueError("The effort_limit must be provided for the delayed DC motor actuator model.")

        # Calculate saturation effort from critical speed
        # From the torque-speed curve: effort_limit = saturation_effort * (1 - critical_speed/velocity_limit)
        # So: saturation_effort = effort_limit / (1 - critical_speed/velocity_limit)
        self._saturation_effort = self.cfg.effort_limit / (1 - self.cfg.critical_speed / self.cfg.velocity_limit)

        # The critical speed is where the torque-speed curve intersects with continuous torque
        self._vel_at_effort_lim = self.cfg.critical_speed

        # Prepare joint velocity buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)

        # Create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)

    def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Save current joint velocity for saturation calculation
        self._joint_vel[:] = joint_vel

        # Apply delay to the control action (handled by parent class)
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip the effort based on the DC motor torque-speed characteristics."""
        # Clip joint velocity to stay within critical speed limits
        self._joint_vel[:] = torch.clip(
            self._joint_vel,
            min=-self._vel_at_effort_lim,
            max=self._vel_at_effort_lim
        )

        # Compute torque limits based on linear DC motor torque-speed curve
        # Top quadrant (positive velocity, positive torque)
        torque_speed_top = self._saturation_effort * (1.0 - self._joint_vel / self.cfg.velocity_limit)
        # Bottom quadrant (negative velocity, negative torque)
        torque_speed_bottom = self._saturation_effort * (-1.0 - self._joint_vel / self.cfg.velocity_limit)

        # Apply continuous torque limits
        max_effort = torch.clip(torque_speed_top, max=self.cfg.effort_limit)
        min_effort = torch.clip(torque_speed_bottom, min=-self.cfg.effort_limit)

        # Clip the computed efforts based on motor limits
        clamped_effort = torch.clip(effort, min=min_effort, max=max_effort)
        return clamped_effort

    @property
    def saturation_effort(self) -> torch.Tensor:
        """Get the calculated saturation effort (stall torque)."""
        return self._saturation_effort

    @property
    def critical_speed(self) -> torch.Tensor:
        """Get the critical speed."""
        return self._vel_at_effort_lim

@configclass
class DelayedDCMotorCfg(DelayedPDActuatorCfg):
    """Configuration for a delayed DC motor actuator."""

    class_type: type = DelayedDCMotor

    critical_speed: float = MISSING
    """Critical speed (rad/s) at which the motor transitions from linear to constant torque behavior."""
