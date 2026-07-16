# Structure-Aware Tensor Information Metrics

This document defines the structure-aware metrics implemented in
`src/NMFkInformationTheory.jl`. Mathematical expressions use standard LaTeX
delimiters and are rendered as text-based mathematical symbols by compatible
Markdown viewers, including GitHub and the VS Code Markdown preview.

## Quantization and adjacent pairs

Let the tensor be quantized into $B$ integer symbols. For every finite, valid
tensor value $x$, the implementation computes

$$
Q(x)=
\mathrm{clamp}\left(
1+\left\lfloor
B\frac{x-x_{\min}}{x_{\max}-x_{\min}}
\right\rfloor,
1,B
\right).
$$

If all valid values are identical, they are assigned symbol $1$.

For tensor axis $d$, let $(X_k,Y_k)$ denote a pair of valid adjacent quantized
cells:

$$
X_k=Q(\mathbf i_k),
\qquad
Y_k=Q(\mathbf i_k+\mathbf e_d).
$$

The corresponding prediction residual is

$$
R_k=Y_k-X_k,
\qquad
R_k\in\{-(B-1),\ldots,B-1\}.
$$

For an empirical discrete variable $X$, entropy in bits is

$$
H(X)=-\sum_x p_X(x)\log_2 p_X(x).
$$

## Bits per temporal residual

For $N_t$ valid adjacent pairs along the temporal axis, bits per temporal
residual is

$$
b_{\mathrm{coded}}=\frac{L_{\mathrm{coded}}}{N_t},
$$

where $L_{\mathrm{coded}}$ is the encoded length of the temporal residual
sequence.

### Fixed-width coding

There are $2B-1$ possible residual symbols. Fixed-width coding therefore uses

$$
b_{\mathrm{fixed}} =
\left\lceil\log_2(2B-1)\right\rceil
$$

bits per residual, with total length

$$
L_{\mathrm{fixed}} =
N_t\left\lceil\log_2(2B-1)\right\rceil.
$$

### Shannon limit

The ideal entropy-coding limit is

$$
b_{\mathrm{Shannon}} = H(R) =
-\sum_r p_R(r)\log_2p_R(r).
$$

This is a theoretical lower bound rather than a concrete code-stream length.

### Huffman coding

If $n_r$ is the observed frequency of residual $r$ and $\ell_r$ is its binary
Huffman codeword length, then

$$
b_{\mathrm{Huffman}} =
\frac{1}{N_t}\sum_r n_r\ell_r =
\sum_r p_R(r)\ell_r.
$$

For a nondegenerate residual alphabet, Huffman coding satisfies

$$
H(R)
\leq
b_{\mathrm{Huffman}}
<
H(R)+1.
$$

When the residual stream contains only one distinct symbol, the implementation
assigns zero payload bits because the symbol is known once the stream metadata is
available.

The plotted **Selected coding** value is controlled by `residual_coding`. The
Oklahoma analysis scripts currently select `:huffman`.

## Spatial dependence

For spatial axis $d$, mutual information between adjacent cells is

$$
I_d(X;Y)=H(X)+H(Y)-H(X,Y).
$$

The implementation normalizes it as

$$
D_d=
\begin{cases}
\dfrac{I_d(X;Y)}{\max\!\left(H(X),H(Y)\right)},
& \max\!\left(H(X),H(Y)\right)>0,\\
0, & \text{otherwise}.
\end{cases}
$$

Let $\mathcal S$ denote the set of axes classified as spatial. Reported spatial
dependence is

$$
D_{\mathrm{spatial}} =
\frac{1}{|\mathcal S|}
\sum_{d\in\mathcal S}D_d.
$$

Interpretation:

- $D_{\mathrm{spatial}}\approx0$: neighboring symbols provide little information
  about one another.
- $D_{\mathrm{spatial}}\approx1$: one neighboring symbol strongly determines the
  other.

Dependence does not necessarily imply smoothness. For example, a deterministic
checkerboard can have high spatial dependence.

## Spatial coherence

For spatial axis $d$, normalized mean adjacent variation is

$$
V_d =
\frac{1}{N_d(B-1)}
\sum_{k=1}^{N_d}|Y_k-X_k|,
$$

where $N_d$ is the number of valid adjacent pairs along that axis.

Mean spatial variation is

$$
V_{\mathrm{spatial}} =
\frac{1}{|\mathcal S|}
\sum_{d\in\mathcal S}V_d.
$$

The summary figure defines spatial coherence as

$$
C_{\mathrm{spatial}}=1-V_{\mathrm{spatial}}.
$$

Interpretation:

- $C_{\mathrm{spatial}}=1$: adjacent spatial symbols are identical.
- $C_{\mathrm{spatial}}\approx0$: adjacent symbols commonly differ by nearly the
  full quantization range.

Unlike spatial dependence, spatial coherence directly measures local smoothness.

## Temporal dependence

Temporal dependence applies the normalized mutual-information definition to
adjacent time bins:

$$
D_{\mathrm{temporal}} =
\begin{cases}
\dfrac{
H(Q_t)+H(Q_{t+1})-H(Q_t,Q_{t+1})
}{
\max\!\left(H(Q_t),H(Q_{t+1})\right)
},
& \max\!\left(H(Q_t),H(Q_{t+1})\right)>0,\\
0, & \text{otherwise}.
\end{cases}
$$

For a single temporal axis, it is available from the axis-level results:

```julia
temporal_axis::NamedTuple = only(filter(
    metric::NamedTuple -> metric.role == :temporal,
    information.axis_information,
))

temporal_dependence::Float64 = temporal_axis.normalized_mutual_information
```

The current structure summary figure plots **temporal coherence**, not temporal
dependence. Temporal coherence is

$$
C_{\mathrm{temporal}} =
1-
\frac{1}{N_t(B-1)}
\sum_{k=1}^{N_t}|Q_{t+1,k}-Q_{t,k}|.
$$

Thus, temporal dependence measures statistical predictability, while temporal
coherence measures direct similarity or lack of change. A deterministic moving or
alternating pattern can have high temporal dependence but lower temporal coherence.

## Residual coding savings

Let

$$
L_{\mathrm{fixed}} =
N_t\left\lceil\log_2(2B-1)\right\rceil
$$

be the fixed-width residual length, and let $L_{\mathrm{coded}}$ be the residual
length produced by the selected Shannon or Huffman model. Residual coding savings
is

$$
S_{\mathrm{coding}} =
\mathrm{clamp}\left(
1-\frac{L_{\mathrm{coded}}}{L_{\mathrm{fixed}}},
0,1
\right).
$$

Before clamping, this is equivalently

$$
S_{\mathrm{coding}} =
\frac{L_{\mathrm{fixed}}-L_{\mathrm{coded}}}
{L_{\mathrm{fixed}}}.
$$

Interpretation:

- $S_{\mathrm{coding}}=0$: no reduction relative to fixed-width residuals.
- $S_{\mathrm{coding}}=0.5$: 50% fewer residual payload bits.
- $S_{\mathrm{coding}}=0.9$: 90% fewer residual payload bits.
- $S_{\mathrm{coding}}=1$: zero residual payload bits under the selected model.

This metric excludes codebook storage, tensor metadata, file headers, and container
overhead. It measures only the modeled residual payload.

## Raw-data information before gridding

`rawdata_information` measures the empirical discrete information in observation
records before they are binned. The name is intentionally domain-neutral: the
same API applies to seismic events, measurements, trajectories, transactions,
sensor samples, or any other row-oriented data.

Continuous values must be interpreted at a declared measurement precision. For
numeric feature $j$, the raw symbol is

$$
Q_j(x)=\left\lfloor\frac{x-o_j}{\delta_j}\right\rfloor,
$$

where $\delta_j>0$ is a fixed precision and $o_j$ is a fixed origin. Date and
time values use the same construction with a fixed `Dates.Period`, such as
`Dates.Second(1)`. Pass `nothing` as the precision for a feature that is already
discrete or categorical.

Floating-point quotients are snapped to an integer boundary only when the
roundoff distance is at most $10^{-9}$ of a bin and the local floating-point
resolution supports that decision; values farther below a boundary remain in
the lower half-open bin. Integer arithmetic is exact, but the resulting bin
coordinate must fit Julia's `Int` range.

`Date` and `DateTime` inputs support fixed periods from `Dates.Week` through
`Dates.Millisecond`. Their finest supported resolution is one millisecond;
calendar periods such as month/year and standalone `Dates.Time` values are not
supported by this encoder.

The origin and precision must remain unchanged across every grid comparison.
Data-dependent minimum/maximum quantization is not used because it would change
the meaning of raw symbols between analyses.

For selected state features, the raw state of observation $i$ is the joint tuple

$$
X_i=\left(Q_1(x_{i1}),\ldots,Q_d(x_{id})\right).
$$

With optional nonnegative observation weights $w_i$, total weight
$W=\sum_i w_i$, and

$$
p_X(x)=\frac{1}{W}\sum_i w_i\,\mathbf{1}(X_i=x),
$$

the raw-state entropy is

$$
H(X)=-\sum_x p_X(x)\log_2 p_X(x).
$$

This is joint entropy. It is generally not equal to the sum of the marginal
feature entropies because raw features can be statistically dependent.

This is the empirical plug-in entropy of the observed sample. It is
sample-dependent and cannot exceed $\log_2N$ for $N$ equally weighted valid
observations. When every fixed-precision row is unique, the state baseline equals
the record-identity baseline; it should not be interpreted as all physical or
semantic information that could exist in the underlying process.

Rows are analyzed on a complete-case basis across all supplied feature columns,
including ancillary columns not selected in `state_features`. Missing or
nonfinite values, a false `valid_mask` entry, and zero-weight rows are excluded.
Positive weights are divided by their largest valid value before storage, because
only their ratios affect the probabilities. The returned `weight_scale` and
`scaled_total_weight` state this normalization explicitly.

The function also returns:

- $2^{H(X)}$, the effective number of raw states;
- observed and singleton state counts;
- marginal information for every feature;
- prediction-residual entropy and selected residual-coding cost for each ordered
  numeric or date/time feature.

Raw residuals are calculated separately in each physical feature. Differences
between arbitrary joint-state identifiers are never used because those identifiers
have no numerical meaning. Residual metrics are unweighted sequence metrics even
when state entropies use nonuniform positive weights. If `sequence_order` is not
supplied, they follow input row order; unordered tables should provide an explicit
ordering such as `sortperm(timestamps)`.

### Empirical distinguishability retained and merged by a grid

`compare_rawdata_grid` compares the same observations in two paired
representations:

- $X$: the fixed-precision raw state;
- $G$: the per-observation grid address, for example
  `(longitude_cell, latitude_cell, time_cell)`.

The implementation calculates

$$
H(X),\qquad H(G),\qquad H(X,G),
$$

and defines retained observed-state distinguishability as mutual information:

$$
I(X;G)=H(X)+H(G)-H(X,G).
$$

Observed-state distinguishability lost when records merge under the same grid
label is the conditional entropy

$$
H(X\mid G)=H(X,G)-H(G).
$$

These satisfy the exact decomposition

$$
H(X)=I(X;G)+H(X\mid G).
$$

When the grid address is a deterministic function of the raw state,
$H(G\mid X)=0$, so the equations simplify to

$$
I(X;G)=H(G),
\qquad
H(X\mid G)=H(X)-H(G).
$$

The intrinsic empirical fractions plotted by `plot_rawdata_grid_information` are

$$
R_{\mathrm{retained}}=\frac{I(X;G)}{H(X)},
\qquad
R_{\mathrm{lost}}=\frac{H(X\mid G)}{H(X)}.
$$

Therefore

$$
R_{\mathrm{retained}}+R_{\mathrm{lost}}=1.
$$

These fractions are not min-max normalized across the tested resolutions. A 90%
retention value means 90% of the stated empirical raw baseline for that sample.
Cross-dataset comparison additionally requires the same feature definition,
precision, origins, weighting, and comparable sampling density and sample size.

This is a merging/collision measure on the observed support. It is not geometric
precision loss, rate-distortion, or the number of bits needed to reconstruct the
original numeric values. A one-to-one assignment of observed raw states to grid
labels has zero merging loss even if the labels do not store within-cell offsets;
an arbitrary permutation of those labels gives the same mutual information. Use
a declared grid-cell reconstruction and analyze its physical residuals when the
question is coordinate/measurement error or lossless reconstruction cost.

If the selected raw baseline has zero entropy, these fractions are undefined and
are returned as `NaN`. The plotting function rejects that state baseline instead
of displaying a misleading 100% retention bar; use a finer raw precision or a
nondegenerate record baseline.

The effective ambiguity

$$
A_{\mathrm{effective}}=2^{H(X\mid G)}
$$

is the effective number of distinguishable raw states collapsed into one grid
state. For unique, equally weighted observations it is the occupancy-weighted
geometric mean number of records merged within a cell.

The diagnostic

$$
H(G\mid X)=H(X,G)-H(X)
$$

is returned as `mapping_uncertainty_bits`. Exact observed-support conflicts are
also reported by `mapping_is_deterministic`, `conflicting_raw_state_count`, and
`mapping_conflict_weight_fraction`. Here “deterministic” means only that each raw
state observed in this sample has one grid label; when every observed raw state is
unique, this condition is automatically true. A conflict warns that the declared
raw quantization is too coarse or misaligned with the grid. This is why the
implementation uses paired mutual information rather than assuming that
$H(G)/H(X)$ is always a valid empirical retention fraction.

Retention is guaranteed to decrease under coarsening only when each coarser grid
is genuinely a nested deterministic partition of the finer grid. Grids with
shifted edges or non-integer changes in cell width need not be nested, so a
non-monotone curve is not automatically an implementation error.

### Raw states versus raw records

The comparison returns two explicitly named baselines:

- **State baseline**: distinguishes observed fixed-precision raw value tuples and
  uses $I(X;G)$.
- **Record baseline**: distinguishes individual input records and uses the grid
  entropy $H(G)$ retained from record identity.

They are equal when every raw state is unique. When exact duplicate raw rows
exist, the state baseline treats them as the same data state, while the record
baseline still distinguishes their record identities.

For unit-weight observations and a count tensor with cell counts $n_g$,

$$
p_G(g)=\frac{n_g}{\sum_h n_h}.
$$

Consequently, `tensor_information(count_tensor).entropy_bits` equals the paired
grid entropy $H(G)`. This equality is checked in the Oklahoma analysis workflow.

Maximum, mean, and median aggregation are not additive pushforwards of
observation mass. In particular, entropy computed from a maximum-magnitude tensor
must not be labeled as raw information retained. Such tensors remain useful
descriptive summaries, but their aggregation loss requires a reconstruction,
distortion, prediction, or explicit coding model.

### Julia example

```julia
import NMFk
import Dates
import Gadfly

raw_information::NamedTuple = NMFk.rawdata_information(
    (
        longitude=longitudes,
        latitude=latitudes,
        time=dates,
        measurement=measurements,
    );
    precisions=(
        longitude=1.0e-4,
        latitude=1.0e-4,
        time=Dates.Second(1),
        measurement=0.1,
    ),
    origins=(
        longitude=0.0,
        latitude=0.0,
        time=Dates.DateTime(1970, 1, 1),
        measurement=0.0,
    ),
    state_features=(:longitude, :latitude, :time),
    sequence_order=sortperm(dates),
    residual_coding=:huffman,
)

comparison::NamedTuple = NMFk.compare_rawdata_grid(
    raw_information,
    (
        longitude=longitude_indices,
        latitude=latitude_indices,
        time=date_indices,
    );
    grid_cell_count=longitude_cell_count * latitude_cell_count * time_cell_count,
)

retention_plot::Gadfly.Plot = NMFk.plot_rawdata_grid_information(
    comparisons,
    resolution_labels,
    "rawdata_information_retention.png";
    normalize=:fraction,
    baseline=:states,
)

loss_heatmap::Gadfly.Plot = NMFk.plot_rawdata_grid_heatmap(
    comparison_matrix,
    spatial_resolution_labels,
    temporal_resolution_labels,
    "rawdata_information_loss_heatmap.png";
    baseline=:states,
    quantity=:lost,
    x_label="Spatial resolution",
    y_label="Temporal resolution",
)
```

The fraction figure uses green for retained observed-state distinguishability,
red for merging loss, and a black empirical raw-data reference line. Each bar is
annotated with retained percentage, merging-loss bits per observation, and
effective ambiguity.
The two-dimensional heatmap uses an absolute zero-to-one color scale. With
`quantity=:lost`, green means little observed-state distinguishability is merged
and red means severe merging; the printed cell values are empirical raw-baseline
merging-loss percentages.

## Oklahoma result persistence and reuse

The Oklahoma structure-analysis workflow writes a version-2 JLD2 checkpoint
before tensor work and after every completed grid configuration. The top-level
`analysis.rawdata_information` stores the raw baseline, precision/origin metadata,
state fingerprint, and feature residual summaries. A separate exact fingerprint
of all selected dates, coordinates, magnitudes, and their ordering prevents a
checkpoint from being resumed against changed measurement values. Every grid
result stores `count_mass`, `rawdata_comparison`, and `grid_shape` alongside the
existing maximum-magnitude and structure summaries.

The Oklahoma retained/lost baseline deliberately uses longitude, latitude, and
time. It therefore measures event-localization information lost by spatial and
temporal binning. Magnitude is still analyzed as a raw feature and in the
maximum-magnitude tensor, but the heatmap does not claim to measure magnitude
aggregation loss. A maximum is a nonadditive summary, so that question requires
an explicitly chosen reconstruction or distortion model rather than an entropy
ratio that would look precise but be misleading.

`run_oklahoma_structure_analysis(...; resume=true)` is the default. It validates
the dataset fingerprint and configuration, skips completed configurations, and
reuses each computed grid for both spatial and temporal figure families. Version-1
checkpoints lack a complete historical input fingerprint and therefore are not
trusted automatically. If, and only if, the current inputs are exactly those used
for that checkpoint, pass `trust_version1_checkpoint=true`; the file is then
backed up with a `.v1.backup` suffix and upgraded by adding sparse raw/grid
comparisons without recomputing its expensive tensor metrics.

The recreated figure set includes
`<dataset>_rawdata_information_loss_heatmap.png`, which places spatial resolution
on one axis, temporal resolution on the other, and directly displays the
percentage of observed raw-state distinguishability lost through grid collisions
for every configuration.

Figures can be recreated from the checkpoint alone:

```julia
import Dates

include(joinpath(
    raw"C:\Users\monty\Julia\Oklahoma.jl\scripts",
    "oklahoma_structure_information_common.jl",
))

analysis::NamedTuple = recreate_oklahoma_structure_figures(
    raw"C:\path\to\dataset_information_theory_results.jld2",
    raw"C:\path\to\new_figure_directory",
)
```
