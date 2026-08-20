# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the ArtiFixer 360 research pipeline by Guilhem Carmouze, 2026.

from pathlib import Path
import unittest


class Artifixer3DDistillationTests(unittest.TestCase):
    def test_release_distillation_defaults_to_scratch30k(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("--artifixer3d_steps", run_source)
        self.assertIn("default=30000", run_source)
        self.assertNotIn("resume_mode", run_source + artifixer3d_source)
        self.assertNotIn("image_path_override_fallback_to_original", run_source + artifixer3d_source)

    def test_resume_requires_explicit_base_checkpoint(self):
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("base_checkpoint: Path | None", artifixer3d_source)
        self.assertIn("if base_checkpoint is not None:", artifixer3d_source)
        self.assertIn('overrides.append(f"resume={base_checkpoint}")', artifixer3d_source)

    def test_distillation_passes_selected_indices_and_image_override(self):
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn('f"selected_indices_file={paths.distillation_selected_indices_path}"', artifixer3d_source)
        self.assertIn('f"image_path_override={paths.override_image_dir.name}"', artifixer3d_source)
        self.assertIn('f"n_iterations={steps}"', artifixer3d_source)

    def test_one_run_can_emit_intermediate_checkpoints(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("--checkpoint_iterations", run_source)
        self.assertIn("checkpoint_iterations=args.checkpoint_iterations", artifixer3d_source)
        self.assertIn('f"checkpoint.iterations=[{checkpoint_iterations_override}]"', artifixer3d_source)

    def test_geometry_locked_mode_uses_fresh_optimizer_and_pixel_losses(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("--geometry_locked", run_source)
        self.assertIn('"resume_optimizer=False"', artifixer3d_source)
        self.assertIn('"model.optimize_position=False"', artifixer3d_source)
        self.assertIn('"model.optimize_rotation=False"', artifixer3d_source)
        self.assertIn('"model.optimize_scale=False"', artifixer3d_source)
        self.assertIn('"model.optimize_density=False"', artifixer3d_source)
        self.assertIn('"loss.lambda_reconlosses_override=1.0"', artifixer3d_source)

    def test_resume_reenables_the_parameter_group_selected_by_the_new_phase(self):
        model_source = Path("thirdparty/3DGRUT-ArtiFixer/threedgrut/model/model.py").read_text()

        self.assertIn(
            "self.features_albedo.requires_grad_(bool(self.conf.model.optimize_features_albedo))",
            model_source,
        )
        self.assertIn(
            "self.density.requires_grad_(bool(self.conf.model.optimize_density))",
            model_source,
        )

    def test_soft_depth_confidence_masks_are_materialized(self):
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn('f"{index:05d}_mask.png"', artifixer3d_source)
        self.assertIn('paths.override_image_dir / f"{index:05d}_mask.png"', artifixer3d_source)

    def test_geometry_refinement_controls_are_forwarded(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        artifixer3d_source = Path("data_processing/artifixer3d.py").read_text()
        trainer_source = Path("thirdparty/3DGRUT-ArtiFixer/threedgrut/trainer.py").read_text()

        self.assertIn("--real_samples_per_override", run_source)
        self.assertIn("--scale_regularization_weight", run_source)
        self.assertIn('f"real_samples_per_override={real_samples_per_override}"', artifixer3d_source)
        self.assertIn('"loss.use_scale=True"', artifixer3d_source)
        self.assertIn("+ lambda_scale * loss_scale", trainer_source)

    def test_opacity_prune_lock_is_monotone_and_disables_topology_changes(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        wrapper_source = Path("data_processing/artifixer3d.py").read_text()
        config_source = Path("thirdparty/3DGRUT-ArtiFixer/configs/base_gs.yaml").read_text()
        model_source = Path("thirdparty/3DGRUT-ArtiFixer/threedgrut/model/model.py").read_text()
        trainer_source = Path("thirdparty/3DGRUT-ArtiFixer/threedgrut/trainer.py").read_text()

        self.assertIn("--opacity_prune_locked", run_source)
        self.assertIn("--density_learning_rate", run_source)
        self.assertIn('"model.optimize_density=True"', wrapper_source)
        self.assertIn('"model.optimize_features_albedo=False"', wrapper_source)
        self.assertIn('"model.monotonic_density_on_resume=True"', wrapper_source)
        self.assertIn("strategy.relocate.start_iteration", wrapper_source)
        self.assertIn("strategy.perturb.start_iteration", wrapper_source)
        self.assertIn("strategy.add.start_iteration", wrapper_source)
        self.assertIn("monotonic_density_on_resume: false", config_source)
        self.assertIn("torch.minimum(self.density, self.density_resume_cap)", model_source)
        self.assertIn("model.project_density_to_resume_cap()", trainer_source)

    def test_opacity_prune_lock_rejects_geometry_lock_and_base_depth(self):
        wrapper_source = Path("data_processing/artifixer3d.py").read_text()
        self.assertIn("geometry_locked and opacity_prune_locked", wrapper_source)
        self.assertIn("opacity_prune_locked and depth_guided", wrapper_source)

    def test_single_surface_mode_requires_external_real_geometry(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        wrapper_source = Path("data_processing/artifixer3d.py").read_text()
        model_source = Path("thirdparty/3DGRUT-ArtiFixer/threedgrut/model/model.py").read_text()
        trainer_source = Path("thirdparty/3DGRUT-ArtiFixer/threedgrut/trainer.py").read_text()

        self.assertIn("--single_surface_guided", run_source)
        self.assertIn("--surface_depth_manifest", run_source)
        self.assertIn("--geometry_authorization", run_source)
        self.assertIn("uses_base_3dgs_depth", wrapper_source)
        self.assertIn("PATCHMATCH_FUSED_REAL_ONLY", wrapper_source)
        self.assertIn('if "3dgut" in config_name.lower()', wrapper_source)
        self.assertIn('outputs["pred_dist_front"]', model_source)
        self.assertIn('loss-single-surface', trainer_source)

    def test_final_plus_has_an_opt_in_fail_closed_geometry_gate(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        wrapper_source = Path("data_processing/artifixer3d.py").read_text()
        self.assertIn("--require_geometry_qc_for_plus", run_source)
        self.assertIn("--geometry_qc_report", run_source)
        self.assertIn("PASS_SINGLE_SURFACE", wrapper_source)
        self.assertIn("ArtiFixer3D+ preparation is forbidden", wrapper_source)

    def test_resume_topology_window_reopens_mcmc_explicitly(self):
        run_source = Path("data_processing/run_artifixer3d.py").read_text()
        wrapper_source = Path("data_processing/artifixer3d.py").read_text()

        self.assertIn("--resume_topology_start_iteration", run_source)
        self.assertIn("--resume_topology_add_relocate_end_iteration", run_source)
        self.assertIn("--resume_topology_perturb_end_iteration", run_source)
        self.assertIn("--resume_topology_max_gaussians", run_source)
        self.assertIn('"model.optimize_position=True"', wrapper_source)
        self.assertIn('"model.optimize_density=True"', wrapper_source)
        self.assertIn("strategy.add.max_n_gaussians", wrapper_source)
        self.assertIn(
            "resumed topology refinement is mutually exclusive with geometry and opacity-prune locks",
            wrapper_source,
        )


if __name__ == "__main__":
    unittest.main()
