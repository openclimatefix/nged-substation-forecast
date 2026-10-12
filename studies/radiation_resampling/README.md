# Radiation resampling illustration

The synthetic example calls the public production feature pipeline and compares it with point
interpolation on identical native interval means. Its reference uses independent pvlib solar
geometry and an invented long-wave curve. The synthetic comparison checks timing and conservation,
not forecast skill or recovery of unknown sub-window changes. The [architecture
explanation](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#radiation-resampling)
includes the figure and its limitations.

Run from the repository root:

```bash
uv run python studies/radiation_resampling/plot_radiation_resampling.py
npx svgo@4 --multipass --precision=1 --final-newline docs/architecture/assets/radiation-resampling.svg
```

The script writes the SVG and PNG in `docs/architecture/assets/` and the numerical checks in
`studies/radiation_resampling/report.md`. Both native step widths use a complete shared middle day;
centroid shifts include all 48 half-hours. Float64 reductions compare the reconstructed values with
the supplied Float32 native means. Assertions stop generation if conservation or timing checks fail.
