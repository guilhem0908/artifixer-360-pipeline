import torch
import torch.nn.functional as F

from model_eval.synchronized_multiview import (
    LatentReprojection,
    build_depth_aware_latent_reprojection_graph,
    build_latent_reprojection_graph,
    perspective_reprojection_grid,
    perspective_reprojection_grid_rotations,
    synchronize_low_frequency_latent_views,
    synchronize_latent_pair,
    synchronize_latent_views,
    validate_reprojection_graph,
    warp_video_latents,
    yaw_rotation,
)


def test_identity_reprojection_preserves_pixel_centres():
    grid, valid = perspective_reprojection_grid(
        24,
        32,
        110.0,
        source_yaw_degrees=60.0,
        target_yaw_degrees=60.0,
        device=torch.device("cpu"),
    )
    image = torch.arange(24 * 32, dtype=torch.float32).view(1, 1, 24, 32)
    warped = warp_video_latents(image, grid)
    torch.testing.assert_close(warped, image)
    assert valid.all()


def test_pair_synchronization_reduces_overlap_disagreement():
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn(2, 4, 3, 48, 48, generator=generator)
    _, initial = synchronize_latent_pair(
        latents,
        fov_degrees=110.0,
        yaw_degrees=(60.0, 120.0),
        blend=0.0,
    )
    _, fused = synchronize_latent_pair(
        latents,
        fov_degrees=110.0,
        yaw_degrees=(60.0, 120.0),
        blend=0.5,
    )
    assert 0.2 < fused["overlap_fraction_view0"] < 0.8
    assert fused["latent_l1_after"] < initial["latent_l1_after"]


def test_rotation_reprojection_matches_legacy_yaw_grid():
    device = torch.device("cpu")
    legacy_grid, legacy_valid = perspective_reprojection_grid(
        48,
        64,
        110.0,
        source_yaw_degrees=60.0,
        target_yaw_degrees=120.0,
        device=device,
    )
    rotation_grid, rotation_valid = perspective_reprojection_grid_rotations(
        48,
        64,
        110.0,
        source_rotation=yaw_rotation(60.0, device=device),
        target_rotation=yaw_rotation(120.0, device=device),
        device=device,
    )
    torch.testing.assert_close(rotation_grid, legacy_grid)
    torch.testing.assert_close(rotation_valid, legacy_valid)


def test_global_consensus_closes_horizontal_and_vertical_overlaps():
    device = torch.device("cpu")
    pitch_up = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 2**-0.5, -(2**-0.5)), (0.0, 2**-0.5, 2**-0.5)),
        dtype=torch.float32,
    )
    rotations = [
        yaw_rotation(0.0, device=device),
        yaw_rotation(60.0, device=device),
        pitch_up,
    ]
    graph = build_latent_reprojection_graph(
        rotations,
        32,
        32,
        110.0,
        device=device,
        dtype=torch.float32,
        minimum_overlap_fraction=0.02,
    )
    generator = torch.Generator().manual_seed(17)
    latents = torch.randn(3, 4, 2, 32, 32, generator=generator)
    _, initial = synchronize_latent_views(latents, graph, blend=0.0)
    synchronized, final = synchronize_latent_views(latents, graph, blend=1.0)
    assert synchronized.shape == latents.shape
    assert final["directed_edges"] >= 4
    assert final["latent_l1_after"] < initial["latent_l1_after"]


def test_low_frequency_consensus_preserves_high_frequency_residual():
    height = width = 16
    identity_grid, valid = perspective_reprojection_grid(
        height,
        width,
        110.0,
        source_yaw_degrees=0.0,
        target_yaw_degrees=0.0,
        device=torch.device("cpu"),
    )
    graph = [
        LatentReprojection(0, 1, identity_grid, valid, 1.0),
        LatentReprojection(1, 0, identity_grid, valid, 1.0),
    ]
    checkerboard = ((torch.arange(height)[:, None] + torch.arange(width)[None, :]) % 2) * 2 - 1
    latents = torch.zeros(2, 1, 1, height, width, dtype=torch.float32)
    latents[0, 0, 0] = checkerboard + 1.0
    latents[1, 0, 0] = -checkerboard - 1.0
    low_frequency = F.avg_pool2d(
        F.pad(latents[:, 0, 0].unsqueeze(1), (2, 2, 2, 2), mode="reflect"),
        kernel_size=5,
        stride=1,
    ).unsqueeze(2)
    synchronized_low, _ = synchronize_latent_views(low_frequency, graph, blend=1.0)
    result, stats = synchronize_low_frequency_latent_views(
        latents,
        graph,
        kernel_size=5,
        blend=1.0,
    )
    assert stats["latent_l1_after"] < stats["latent_l1_before"]
    assert stats["high_frequency_residual_max_error"] == 0.0
    torch.testing.assert_close(result - synchronized_low, latents - low_frequency)


def test_spatiotemporal_validity_never_changes_occluded_frames():
    latents = torch.zeros(2, 1, 2, 8, 8)
    latents[0] = 1
    grid, spatial = perspective_reprojection_grid(
        8,
        8,
        110.0,
        source_yaw_degrees=0.0,
        target_yaw_degrees=0.0,
        device=torch.device("cpu"),
    )
    valid = torch.stack([spatial, torch.zeros_like(spatial)])
    graph = [LatentReprojection(0, 1, grid, valid, float(valid.float().mean()))]
    synchronized, _ = synchronize_latent_views(latents, graph, blend=1.0)
    assert torch.count_nonzero(synchronized[1, :, 0]) > 0
    assert torch.count_nonzero(synchronized[1, :, 1]) == 0


def test_depth_aware_graph_keeps_only_depth_consistent_visible_rays():
    device = torch.device("cpu")
    rotations = [yaw_rotation(0.0, device=device), yaw_rotation(60.0, device=device)]
    depths = torch.full((2, 5, 32, 32), 2.0)
    opacity = torch.ones_like(depths)
    graph = build_depth_aware_latent_reprojection_graph(
        rotations,
        depths,
        16,
        16,
        110.0,
        device=device,
        dtype=torch.float32,
        opacity=opacity,
        minimum_depth_overlap_fraction=0.001,
    )
    assert len(graph) == 2
    assert all(edge.valid.shape == (2, 16, 16) for edge in graph)
    stats = validate_reprojection_graph(
        graph,
        ["H000", "H060"],
        required_bidirectional_pairs=[("H000", "H060")],
    )
    assert stats["minimum_depth_valid_fraction"] > 0

    inconsistent = depths.clone()
    inconsistent[1] = 8.0
    try:
        build_depth_aware_latent_reprojection_graph(
            rotations,
            inconsistent,
            16,
            16,
            110.0,
            device=device,
            dtype=torch.float32,
            opacity=opacity,
            minimum_depth_overlap_fraction=0.001,
        )
    except ValueError as error:
        assert "removed every" in str(error)
    else:
        raise AssertionError("contradictory radial depths must fail closed")
