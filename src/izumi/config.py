"""Structured config dataclasses for Hydra configuration."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class GPTConf:
    block_size: int = 50
    input_dim: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 120


@dataclass
class EncoderConf:
    _target_: str = "izumi.models.encoder.TimmEncoder"
    model_name: str = "hf-hub:notmahi/dobb-e"


@dataclass
class VQBeTConf:
    _target_: str = "izumi.models.policy.VQBeTPolicy"
    action_dim: int = 7
    obs_window_size: int = 3
    action_sequence_length: int = 1
    vqvae_n_latent_dims: int = 512
    vqvae_n_embed: int = 16
    vqvae_groups: int = 2
    vqvae_load_dir: str | None = None
    temperature: float = 1e-6
    sequentially_select: bool = True
    offset_loss_multiplier: float = 10.0
    secondary_code_multiplier: float = 0.5
    gamma: float = 2.0
    gpt: GPTConf = field(default_factory=GPTConf)


@dataclass
class DiffusionConf:
    _target_: str = "izumi.models.diffusion_policy.DiffusionPolicy"
    action_dim: int = 7
    obs_dim: int = 512
    obs_window_size: int = 6
    action_sequence_length: int = 6
    num_extra_actions: int = 5
    policy_type: str = "transformer"
    data_act_scale: float = 1.0
    data_obs_scale: float = 1.0


@dataclass
class PortsConf:
    status: int = 5555
    command: int = 5556
    servo: int = 5558
    d405: int = 6002


@dataclass
class RobotConf:
    host: str = "127.0.0.1"
    ports: PortsConf = field(default_factory=PortsConf)
    camera: str = "d405"
    control_hz: float = 5.0


@dataclass
class InferenceVQBeTConf:
    defaults: list = field(
        default_factory=lambda: [
            {"model": "resnet_dobbe"},
            {"policy": "vqbet"},
            {"robot": "stretch3"},
            "_self_",
        ]
    )
    task: str = MISSING
    device: str = "cuda"
    image_buffer_size: int = 3
    checkpoint: str = MISSING
    model: EncoderConf = field(default_factory=EncoderConf)
    policy: VQBeTConf = field(default_factory=VQBeTConf)
    robot: RobotConf = field(default_factory=RobotConf)


@dataclass
class InferenceDiffusionConf:
    defaults: list = field(
        default_factory=lambda: [
            {"model": "resnet_dobbe"},
            {"policy": "diffusion"},
            {"robot": "stretch3"},
            "_self_",
        ]
    )
    task: str = MISSING
    device: str = "cuda"
    image_buffer_size: int = 6
    checkpoint: str = MISSING
    model: EncoderConf = field(default_factory=EncoderConf)
    policy: DiffusionConf = field(default_factory=DiffusionConf)
    robot: RobotConf = field(default_factory=RobotConf)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="inference_vqbet_schema", node=InferenceVQBeTConf)
    cs.store(name="inference_diffusion_schema", node=InferenceDiffusionConf)
    cs.store(group="model", name="resnet_dobbe_schema", node=EncoderConf)
    cs.store(group="policy", name="vqbet_schema", node=VQBeTConf)
    cs.store(group="policy", name="diffusion_schema", node=DiffusionConf)
    cs.store(group="robot", name="stretch3_schema", node=RobotConf)
