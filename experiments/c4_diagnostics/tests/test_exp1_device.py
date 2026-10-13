import pytest
import torch

from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, render_synthetic_observation


def _frame(device: torch.device) -> GeometryFrame:
    return GeometryFrame("SAX", "device", torch.zeros(3, device=device), torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor([0.0, 1.0, 0.0], device=device), torch.tensor([0.0, 0.0, 1.0], device=device), torch.tensor([1.5, 1.5], device=device), 5.0, .25, 8, 8)


def test_cpu_phantom_forward_and_volume_stay_on_cpu() -> None:
    phantom = AnalyticCanonicalPhantom().to("cpu")

    values = phantom(torch.zeros(1, 2, 3))
    axes = phantom.volume(torch.tensor([-4.0, -4.0, -4.0]), torch.tensor([4.0, 4.0, 4.0]), (3, 3, 3))

    assert values.device.type == "cpu"
    assert all(array.shape[0] == 3 for array in axes[:3])
    assert axes[-1].shape == (3, 3, 3)


def test_synthetic_render_keeps_every_observation_tensor_on_render_device() -> None:
    device = torch.device("cpu")
    observation = render_synthetic_observation(_frame(device), height=8, width=8, phantom=AnalyticCanonicalPhantom().to(device), motion=GroundTruthMotion(), psf_samples=2, seed=4, dynamic_frame_id=0)

    assert {value.device for value in (observation.image, observation.center_mm, observation.row_direction, observation.column_direction, observation.normal, observation.pixel_spacing_mm)} == {device}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_phantom_forward_and_volume_are_device_safe() -> None:
    device = torch.device("cuda")
    phantom = AnalyticCanonicalPhantom().to(device)

    values = phantom(torch.zeros(1, 2, 3, device=device))
    axes = phantom.volume(torch.tensor([-4.0, -4.0, -4.0]), torch.tensor([4.0, 4.0, 4.0]), (3, 3, 3))

    assert values.device == device
    assert axes[-1].shape == (3, 3, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_synthetic_render_restores_rng_state() -> None:
    device = torch.device("cuda")
    torch.cuda.manual_seed(913)
    before = torch.cuda.get_rng_state(device)

    observation = render_synthetic_observation(_frame(device), height=8, width=8, phantom=AnalyticCanonicalPhantom().to(device), motion=GroundTruthMotion(), psf_samples=2, seed=8, dynamic_frame_id=0)

    assert observation.image.device == device
    assert torch.equal(torch.cuda.get_rng_state(device), before)
