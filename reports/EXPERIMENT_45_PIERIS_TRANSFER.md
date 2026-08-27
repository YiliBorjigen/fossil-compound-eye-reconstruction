# Experiment 45: independent Pieris transfer

## Verdict

The population-prior reconstruction found in one *Apis mellifera* eye did not
transfer to *Pieris napi*. This is a failed independent replication, not a new
positive reconstruction result.

## Evidence chain

The uploaded raw archive contains 992 unsigned 16-bit TIFF slices of 368 × 496
pixels. The associated NIfTI is a 366-slice crop containing the full eye-volume
and external-cornea labels. Maximising the CT gradient at the labelled corneal
surface placed the crop at raw-slice offset 550; offsets 549 and 551 were the
nearest alternatives and had distinctly lower scores.

Three regions were selected automatically from the labelled surface before
reconstruction. Their seeds were separated by at least 274 voxels. Local
unfolding produced a clear periodic crystalline-cone pattern, so the negative
result is not caused by absence of visible cone anatomy.

The exact Experiment 44 surface detector found zero centres at depth zero in
all three regions because it used an absolute threshold inherited from the
8-bit Apis scan. Strict transfer therefore failed before a target could be
scored. A pre-scoring imaging adapter replaced that scanner-specific step with
a shallow-lattice maximum detector and selected the internal cross-section by
lattice regularity. It did not use held-out reconstruction error.

The adapted test retained 35, 21 and 6 matched cones in the three regions. In
15 spatially held-out blocks, the global template beat the depth-matched
background in only four. Pooled over 62 cones, median normalised MAE was 0.176
for background and 0.204 for the global template; median correlation was 0.670
and 0.509, respectively. The median cone-level error increased by 17.9%.

Two post-hoc checks were kept separate from the primary result. Selecting
translation versus affine mapping using training cones only did not restore the
population advantage. Restricting evaluation to a four-voxel central core also
did not restore it.

## Interpretation

The Apis template result was real within that eye, but it is not a transferable
method as currently formulated. A depth-matched background captures much of
the Pieris signal, while template placement and regional variation add error.
This is evidence against generalisation, not evidence that crystalline cones
cannot be detected: the unfolded Pieris volume shows them directly.

The next method should predict residual structure beyond the depth profile,
align units by local axis, scale and rotation, and lock all choices before a
new specimen is examined. Further tuning on these same Apis and Pieris volumes
would not constitute independent validation.
