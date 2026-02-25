Download & process numerical weather predictions from Dynamical.org.

We convert the ECMWF ENS 0.25 degree data to these H3 resolution 5 hexagons:

![Map of Great Britain using H3 resolution 5 hexagons](map-of-Great-Britain-H3-resolution-5.png)

## Data storage experiments

All these experiments were performed on a single model run of ECMWF ENS (2026-02-23T00), just for Great Britain.

Saving a single ECMWF ENS run using `float32`, and `zstd` compression (with default compression
level) results in Parquet files ranging between about 205 MB to 220 MB.

### Test different sort orders:
(After scaling to `[0, 255]` and saving as `UInt8`, and compressing using zstd with the default level)

```
"init_time", "lead_time", "ensemble_member", "h3_index" = 54 MB (BEST YET)
"init_time", "ensemble_member", "lead_time", "h3_index" = 56 MB
"init_time", "lead_time", "h3_index", "ensemble_member" = 59 MB
"init_time", "h3_index", "lead_time", "ensemble_member" = 60 MB
"init_time", "ensemble_member", "h3_index", "lead_time" = 62 MB
```

### Test compression algorithm
(after sorting by "init_time", "lead_time", "ensemble_member", "h3_index", and scaling to `[0, 255]`
and saving as `UInt8`)

```
compression="zstd", compression_level=12 = 54 MB
compression="zstd", compression_level=13 = 53 MB
compression="zstd", compression_level=14 = 51 MB, 2.26s (BEST MIX OF SPEED & COMPRESSION RATIO)
compression="zstd", compression_level=15 = 51 MB
compression="zstd", compression_level=20 = 51 MB
compression="zstd", compression_level=22 = 51 MB
compression="lz4" = 68 MB
compression="snappy" = 78 MB
compression="gzip" = 56 MB
compression="gzip", compression_level=9 = 53 MB, 1.49s
compression="brotli", compression_level=6  = 54 MB, 1.88s
compression="brotli", compression_level=8  = 54 MB, 2.75s
compression="brotli", compression_level=9  = 53 MB, 3.9s
compression="brotli", compression_level=10 = 48 MB, 12.59s
compression="brotli", compression_level=11 = 48 MB, 17.75s!
```

### Testing different dtypes
(after sorting by "init_time", "lead_time", "ensemble_member", "h3_index", and compressing using zstd level 14)

```
Scale to 2¹⁶ - 1, and save as UInt16 = 145 MB
Scale to 2¹⁰ - 1, and save as UInt16 =  79 MB
Scale to 2⁹  - 1, and save as UInt16 =  62 MB
Scale to 2⁸  - 1, and save as UInt16 =  51 MB
Scale to 2⁸  - 1, and save as UInt8  =  51 MB
```
