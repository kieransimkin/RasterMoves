# RasterMoves evaluation checklist

Use this checklist when comparing outputs, deciding whether a result is acceptable,
or documenting a visual improvement. Select only checks relevant to the image.

## Evidence to retain

- Input and output SHA-256, pixel dimensions, format, colour mode and alpha presence.
- RasterMoves version/commit, model ID and licence, backend, device, precision, tiling,
  final resize, refiner/seed/mask settings, report path and trace path when used.
- A note identifying whether weights were newly downloaded, cache-verified, or used
  offline. Do not retain tokens or private cache paths in public documentation.

## Visual checks

- Full-frame composition, crop, orientation, transparency and colour consistency.
- Hard edges and diagonals: halos, stair-stepping, ringing and doubled contours.
- Flat areas and gradients: banding, texture hallucination and tile boundaries.
- Fine texture: plausible detail without waxiness, repeated patterns or oversharpening.
- Text, logos and interfaces: exact legibility; reject invented or mutated glyphs.
- Faces, hands and identity cues: no feature drift. Refinement needs explicit scrutiny
  because it synthesises rather than recovers detail.
- Alpha boundaries: no dark fringe, opaque pixels, colour bleed or unexpected flattening.

Inspect at fit-to-frame and 100%. Use the same crop coordinates and viewing scale for
comparisons. A higher-resolution file is not automatically a better-looking result.

## README example pattern

Show a reproducible command or short API example with named assumptions:

```powershell
rastermoves upscale .\input.png -o .\output.png `
  -m 4x-realesr-general-x4v3 --width 3000 --report
```

Then state the expected dimensions and sidecar, the content type for which the example
was checked, and any licence, memory, or detail-invention limit that matters.

## Screenshot decision

Add a screenshot when readers need to see a seam fix, halo reduction, alpha correction,
model trade-off, or new visual workflow. Prefer a compact labelled before/after image
plus one or two equal-coordinate 100% crops. Store public documentation media under
`docs/assets/` with descriptive names and reference it from `README.md` using meaningful
alt text. Include the command and settings that produced it.

Do not add a screenshot for text-only documentation, an invisible internal refactor,
or a result whose source image cannot safely be published. In those cases, use a
synthetic fixture, test assertion, or concise terminal excerpt instead.
