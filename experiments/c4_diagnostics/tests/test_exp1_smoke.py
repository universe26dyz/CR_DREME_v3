import torch

from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.sampler import DynamicObservation


def test_tiny_three_view_cpu_batch_renders_finite_loss_and_backward() -> None:
    """Would fail if the diagnostic's synthetic slice contract cannot drive a differentiable C4-style loss."""
    torch.manual_seed(7)
    model = SourceFirstDynamicModel(torch.tensor([-12.0, -12.0, -12.0]), torch.tensor([12.0, 12.0, 12.0]), cardiac_lower_world_mm=torch.tensor([-7.0, -7.0, -7.0]), cardiac_upper_world_mm=torch.tensor([7.0, 7.0, 7.0]), n_dynamic_frames=3, inr_width=4, inr_depth=1, latent_dim=4, motion_hidden_dim=4, respiratory_grid_shapes=((2, 2, 2), (2, 2, 2), (2, 2, 2)), cardiac_grid_shape=(2, 2, 2), psf_samples=1, canonical_coarsest_resolution=8.0, canonical_finest_resolution=4.0, canonical_features_per_level=2, canonical_log2_hashmap_size=4, film_channels=4)
    axes = (("SAX", (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), ("2CH", (1., 0., 0.), (0., 0., 1.), (0., -1., 0.)), ("4CH", (0., 1., 0.), (0., 0., 1.), (1., 0., 0.)))
    losses = []
    for frame_id, (view, row, column, normal) in enumerate(axes):
        observation = DynamicObservation(image=torch.rand(1, 4, 4), view=view, slice_id=view, dynamic_frame_id=frame_id, center_mm=torch.zeros(3), row_direction=torch.tensor(row), column_direction=torch.tensor(column), normal=torch.tensor(normal), pixel_spacing_mm=torch.tensor([1.5, 1.5]), slice_thickness_mm=4.0, qc_valid=True, qc_reason="synthetic_gt", timestamp_s=.1 * frame_id)
        pixel_uv = torch.tensor([[0., 0.], [2., 1.]])
        prediction = model.predict(observation, pixel_uv, "stage3b")["predicted_intensity"]
        losses.append((prediction - observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]).square().mean())
    loss = torch.stack(losses).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
