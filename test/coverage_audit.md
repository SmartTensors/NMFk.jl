# NMFk test coverage audit (token-based)

Generated: 2026-01-30

## What this is
This is a quick, *static* audit of functions in `src/` that are **likely not covered by tests**.

Method:
1. Collect function names that appear as `function NAME...` in `src/**/*.jl`.
2. Mark a function as “tested” if the token `NAME` appears anywhere in `test/**/*.jl`.

Summary (current repo state):
- Functions found via `function NAME` in `src/`: **262**
- Not referenced anywhere in `test/` (token match): **228**

## Limitations (important)
- This does **not** detect one-liner definitions like `foo(x) = ...`.
- “Token match” can yield **false positives** (e.g., a variable named `save` in a test) and **false negatives** (e.g., tests calling `NMFk.foo` via indirect dispatch or string-based APIs).
- Private/internal helpers (leading `_`) are included; it’s fine to leave many of those untested.

## Currently referenced (sanity check)
These function names from `src/` *do* appear in `test/` by token match (some may still be incidental):

- `execute`, `mixmatchdata`
- `griddata`, `processdata`, `indicize`
- `r2`, `findfirst`, `movingwindow`, `nanmask!`, `remask`, `flip`, `bincount`
- NaN-aware reducers/metrics: `maximumnan`, `minimumnan`, `sumnan`, `meannan`, `cumsumnan`, `varnan`, `stdnan`, `rmsenan`, `l1nan`, `ssqrnan`, `covnan`, `cornan`, `sortnan`, `sortpermnan`
- Encoding: `hardencodelength`, `hardencode`, `harddecode`

## Likely untested functions (228)
### Internal/private (leading `_`)
- `_blend_pixel`
- `_blend_with_background`
- `_build_geojson_tiles`
- `_cleanup_movie_frames`
- `_colorbar_tick_labels`
- `_colorbar_tick_values`
- `_convert_with_background`
- `_expand_polygon_vertices`
- `_find_frame_files`
- `_flatten_image_to_jpg`
- `_format_decimal_label`
- `_format_scientific_label`
- `_grid_edges`
- `_materialize_frame_sequence`
- `_movie_command_args`
- `_point_between`
- `_point_in_polygon`
- `_polygon_self_intersects`
- `_prepare_hull_points`
- `_remove_files`
- `_resolve_color_bounds`
- `_resolve_recoup_fillvalue`
- `_run_ffmpeg`
- `_segments_intersect`
- `_sort_frame_files!`

### Non-underscore functions
- `aisnan`
- `aisnan!`
- `arp`
- `arp_eur`
- `arp_eur_exponential`
- `arp_eur_harmonic`
- `arp_exponential`
- `arp_harmonic`
- `arrayminmax`
- `bincoordinates`
- `biplot`
- `biplots`
- `bootstrapping`
- `bootstrapping!`
- `branches`
- `build_compression_result`
- `build_scatter_trace`
- `check_traces`
- `checkarray`
- `checkarrayentries`
- `checkmatrix`
- `checkmatrix_robust`
- `checkrectbin`
- `checkvector`
- `cluster_rows`
- `clustersolutions`
- `clustersolutions_old`
- `colormap`
- `colorscale`
- `compress_rows`
- `compute_concave_hull_vertices`
- `compute_convex_hull_vertices`
- `compute_dot_size`
- `compute_zoom`
- `compute_zoom_dot_size`
- `computedeltas`
- `dataframe_rename!`
- `datanalytics`
- `decompress_rows`
- `denormalize`
- `denormalize!`
- `denormalizearray`
- `denormalizearray!`
- `denormalizematrix`
- `denormalizematrix_col`
- `denormalizematrix_col!`
- `denormalizematrix_row`
- `denormalizematrix_row!`
- `denormalizematrix!`
- `descalearray!`
- `descalematrix!`
- `df2matrix`
- `df2matrix_shifted`
- `ensure_mapbox_token!`
- `estimateflip`
- `estimateflip_permutedims`
- `evaluate_compression`
- `execute_run`
- `execute_singlerun`
- `execute_singlerun_compute`
- `fill_nan_with_means!`
- `finalize`
- `finduniquesignals`
- `finduniquesignalsbest`
- `firstjump`
- `fixmixers!`
- `flatten`
- `flattenindex`
- `flip!`
- `Gadfly`
- `get_lonlat`
- `getdatawindow`
- `getk`
- `getks`
- `getmissingattributes`
- `gettypes`
- `grid_reduction`
- `haversine`
- `histogram`
- `idw_interpolate`
- `inferperm`
- `input_checks`
- `ispkgavailable`
- `joinpathcheck`
- `jump`
- `jumpHrows`
- `jumpiter`
- `labelassignements`
- `latin_hypercube_points`
- `latlon_to_xy`
- `layout_fig`
- `load`
- `log10s`
- `log10s!`
- `make_progressbar_2d`
- `makemovie`
- `mapbox`
- `mapbox_colorbar_attr`
- `mapbox_contour`
- `mapping`
- `mapping_old`
- `mapping_permutedims`
- `maskvector`
- `matrixmax`
- `matrixmin`
- `matrixminmax`
- `medoid_index`
- `minmax_dx`
- `mixmatchcompute`
- `mixmatchdeltas`
- `mixmatchwaterdeltas`
- `moviehstack`
- `movievstack`
- `moving_average`
- `NMFmultiplicative`
- `NMFmultiplicativemovie`
- `NMFpreprocessing!`
- `NMFrun`
- `NMFsparsity`
- `normalize`
- `normalize!`
- `normalizearray`
- `normalizearray!`
- `normalizematrix`
- `normalizematrix_col`
- `normalizematrix_col!`
- `normalizematrix_row`
- `normalizematrix_row!`
- `normalizematrix!`
- `normnan`
- `pkginstalled`
- `plot_dots`
- `plot_heel_toe`
- `plot_heel_toe_bad`
- `plot_signal_selecton`
- `plot_wells`
- `plot2dmatrixcomponents`
- `plotbars`
- `plotdendrogram`
- `plotly_layout`
- `plotly_title_length`
- `plotlymatrix`
- `plotmap`
- `plotmatches`
- `plotmatrix`
- `plotscatter`
- `postprocess`
- `prepare_matrix_clustering`
- `printerrormsg`
- `processdata!`
- `progressbar_2d`
- `progressbar_regular`
- `progressive`
- `quietoff`
- `quieton`
- `r2matrix`
- `random_points`
- `recoupmatrix_rows`
- `regression`
- `remap`
- `remap2count`
- `resolve_location_labels`
- `restartoff`
- `restarton`
- `robustkmeans`
- `safe_savefig`
- `sankey`
- `scalearray!`
- `scalematrix_col!`
- `scalematrix_row!`
- `set_typecolors`
- `setbadmixerelements!`
- `setdpi`
- `shiftarray`
- `showsignals`
- `signal_statistics`
- `signalassignments`
- `signalorder`
- `signalorderassignments`
- `signalrescale!`
- `slopes`
- `smoothedzscore`
- `sortclustering`
- `stackmovie`
- `stderrcaptureoff`
- `stderrcaptureon`
- `stdoutcaptureoff`
- `stdoutcaptureon`
- `stdouterrcaptureoff`
- `stdouterrcaptureon`
- `stringfix`
- `stringproduct`
- `style_mapbox_traces!`
- `subset`
- `tensorfactorization`
- `trace_fig`
- `uncertainty`
- `uncertaintyranges`
- `uniform_points`
- `welcome`
- `xy_to_latlon`
- `zerostoepsilon`
- `zerostoepsilon!`

## Suggested next test targets (high-value, low-flake)
If the goal is *unit* coverage without pulling in plotting / Mapbox / ffmpeg, I’d prioritize these:

1. **Core orchestration (lightweight paths)**: `execute_singlerun`, `execute_run`, `finalize`, `input_checks`
2. **Normalization utilities**: `normalize`, `normalize!`, `denormalize`, `denormalize!`, `zerostoepsilon`, `zerostoepsilon!`
3. **Clustering utilities**: `robustkmeans`, `clustersolutions`, `prepare_matrix_clustering`
4. **Compression**: `compress_rows`, `decompress_rows`, `evaluate_compression`, `build_compression_result`
5. **Uncertainty / restart toggles**: `uncertainty`, `uncertaintyranges`, `restarton`, `restartoff`
6. **Signal bookkeeping**: `signal_statistics`, `signalorder`, `signalassignments`

If you want, I can turn this into a set of new test files (e.g. `test_execute_light.jl`, `test_normalize.jl`, `test_cluster.jl`) and keep them deterministic + fast.
