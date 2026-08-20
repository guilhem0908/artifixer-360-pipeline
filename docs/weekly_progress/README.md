# Weekly progress archive

This directory preserves the technical progression of the Gauvain/ArtiFixer-360 internship work as searchable Markdown. It covers every dated weekly/progress artifact found in the project Drive and on the workstation that belonged to this workstream.

The original presentations are large—approximately 15–75 MB each—and together exceed 350 MB. To keep this public Git repository practical to clone, the archive stores their technical text, tables, conclusions and provenance here while linking to the originals on Google Drive. Visual figures and videos remain in those source files and may require Drive permission.

## Timeline

| Date | Milestone | Archive |
|---|---|---|
| 2026-05-11 | DISCOVERSE 360° dataset generation, cubemap-to-ERP baseline and runtime | [Notes](2026-05-11.md) |
| 2026-05-18 | Video → COLMAP → Splatfacto baseline and off-trajectory problem | [Notes](2026-05-18.md) |
| 2026-05-25 | MASt3R/3R-GS/LongSplat reconstruction comparison | [Notes](2026-05-25.md) |
| 2026-06-01 | Survey of 3D world reconstruction and generation methods | [Notes](2026-06-01.md) |
| 2026-06-08 | HY-World 2.0, Habitat-GS and navigation direction | [Notes](2026-06-08.md) |
| 2026-06-15 | NoMaD test and distance-from-trajectory benchmark design | [Notes](2026-06-15.md) |
| 2026-06-22 | Reconstruction/generation trade-off and exploratory benchmark | [Notes](2026-06-22.md) |
| 2026-06-29 | Reference-free metrics, NavGSim and ArtiFixer survey | [Notes](2026-06-29.md) |
| 2026-07-28 | Failure analysis and geometry-first plan | [Notes](2026-07-28.md) |
| 2026-08-03 | Full framework and separation of faithful versus plausible paths | [Notes](2026-08-03.md) |
| 2026-08-09 | Complete fourteen-direction run report and QC failure verdict | [Notes](2026-08-09.md) |
| 2026-08-18 | Raw 3DGRUT versus ArtiFixer3D+, depth/loop comparison and final framework | [Notes](2026-08-18.md) |

## What changed over the internship

The work moved through four phases:

1. **Panorama generation:** render six directions and project them to ERP.
2. **Off-trajectory reconstruction:** compare COLMAP/Splatfacto with learned-pose and long-video alternatives.
3. **Evaluation and navigation:** define distance-from-trajectory metrics, occupancy extraction and navigation constraints.
4. **ArtiFixer-360:** replace independent-face repair with overlapping directions, temporal context, spherical-snake ordering, 3D feedback, coverage checks and exact real-pixel reinsertion.

The final lesson is negative but actionable: appearance restoration and complete spherical coverage do not guarantee one coherent 3D surface. The fourteen-direction run passed coverage and exact source-pixel reinsertion but failed visual, seam and temporal QC. Future work should verify each proposed repair against canonical geometry before committing it to the shared scene.

## Curation policy

Included:

- the complete dated presentation series for this project;
- later progress reviews that superseded the weekly slide naming convention;
- technical tables, failure evidence, conclusions and original Drive links.

Excluded:

- duplicate exports of the same deck;
- administrative and signed documents;
- reports belonging to other interns;
- raw videos or slide binaries whose size would make the public Git history impractical.

The editable final framework itself is versioned separately in the [framework archive](../framework/README.md).
