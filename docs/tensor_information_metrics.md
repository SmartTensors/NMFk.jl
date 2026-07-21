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

For sparse nonnegative count or mass fields,
`quantization=:zero_preserving` reserves symbol $1$ for exact zero and assigns
positive values to symbols $2,\ldots,B$. Let $x_+$ and $x^+$ be the minimum and
maximum positive values. When $x_+<x^+$,

$$
Q_0(x)=
\begin{cases}
1, & x=0,\\
\mathrm{clamp}\left(
2+\left\lfloor
(B-1)\dfrac{x-x_+}{x^+-x_+}
\right\rfloor,
2,B
\right), & x>0.
\end{cases}
$$

If all positive values are identical, they receive symbol $2$. An all-zero
field uses only symbol $1$. Negative valid values are rejected in this mode;
the default `:linear` mode remains available for signed tensors. Reserving zero
prevents a common one-event cell from sharing the empty-cell symbol merely
because a rare cell has a much larger event count.

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

## Multidimensional lag vectors

Axis-adjacent pairs are the special case of a general integer lag vector

$$
\boldsymbol{\delta}=(\delta_1,\ldots,\delta_D)\in\mathbb Z^D,
\qquad
\boldsymbol{\delta}\ne\mathbf 0.
$$

For each grid index $\mathbf i_k$ for which both endpoints are in bounds and
valid, the lagged pair and prediction residual are

$$
X_k=Q(\mathbf i_k),
\qquad
Y_k=Q(\mathbf i_k+\boldsymbol{\delta}),
\qquad
R_k=Y_k-X_k.
$$

This definition includes axial, diagonal, and mixed space-time relationships.
For a tensor ordered as latitude, longitude, time, for example,
$(1,1,0)$ is a spatial diagonal and $(1,-1,2)$ is a directional
space-time lag. For a future latitude, longitude, depth, time tensor, the same
definition extends directly to $(\delta_{\rm lat},\delta_{\rm lon},
\delta_{\rm depth},\delta_t)$.

For that 4-D case, pass
`dimension_roles=(:horizontal, :horizontal, :depth, :temporal)`. Pure depth
lags are then reported with `role=:depth`, horizontal-depth diagonals with
`role=:spatial_depth`, and mixed time lags with `role=:spatiotemporal`.
`depth_dependence` and `depth_variation` remain separate from the horizontal
`spatial_dependence` and `spatial_variation` summaries.

The lag API is dimension-generic, but `structure_information` still receives a
dense tensor. A fine 4-D event grid can therefore run out of memory before lag
histograms are evaluated. The Oklahoma workflow should not add a depth axis to
its present dense tensors without first moving cell aggregation to a truly
sparse occupied-cell representation or deliberately coarsening the grid.

The number of geometrically possible pairs before masking is

$$
N_{\rm candidate}(\boldsymbol{\delta})=
\prod_{d=1}^{D}\max\left(n_d-|\delta_d|,0\right),
$$

where $n_d$ is the size of tensor dimension $d$. The reported pair count can be
smaller because a pair is used only when both endpoints are valid.
This endpoint rule is stored as `lag_pair_scope=:both_endpoints_valid`. With the
default all-valid mask, numerical zeros are valid cells and zero-zero pairs are
included. For a sparse event-count field this describes full-field occupancy
persistence, including quiet background; it is not the same as conditioning on
two occupied event cells.

Lag components remain separate in stored results and plots. A single Euclidean
norm should not combine longitude/latitude degrees, depth units, and elapsed
time. If radial spatial pooling is desired, it must first use an explicitly
chosen compatible coordinate metric; time should remain a separate lag.
The returned `grid_index_norm` is only the dimensionless norm of integer cell
offsets and is not presented as a physical distance.

For a latitude, longitude, time tensor, a compact direction-resolved analysis
can be run as follows:

```julia
import Dates
import Gadfly
import NMFk

lag_offsets::Vector{Tuple{Int,Int,Int}} = [
    (1, 0, 0), (0, 1, 0),
    (1, 1, 0), (1, -1, 0),
    (2, 0, 0), (0, 2, 0),
    (0, 0, 1), (0, 0, 7),
    (1, 1, 1), (1, -1, 1),
    (-1, 1, 1), (-1, -1, 1),
]

information::NamedTuple = NMFk.structure_information(
    flattened_T;
    temporal_dim=3,
    lag_offsets=lag_offsets,
    dimension_names=(:latitude, :longitude, :time),
    dimension_steps=(latitude_cell_degrees, longitude_cell_degrees, Dates.Day(1)),
    dimension_units=("deg", "deg", ""),
    dimension_roles=(:horizontal, :horizontal, :temporal),
    compute_spectral=false,
)

lag_plot::Gadfly.Plot = NMFk.plot_lag_information(information)
```

Floating-point coordinate offsets in lag-plot labels use four significant
digits by default, while the exact values remain stored in
`lag_information.coordinate_offset`. Use
`coordinate_sigdigits=3` for an even more compact display. This formatting is
applied when the plot is rendered, so plots recreated from older checkpoints
also receive the shorter labels.

Use `horizontal_lag_labels=:cells` to display horizontal dimensions as grid-cell
counts, for example `lat=1` and `lon=-2`. Time and depth remain in their physical
coordinate units. This changes only plot labels; stored coordinate offsets and
all information metrics remain unchanged.

When at least one requested lag is applicable, every unavailable lag is displayed
as `NA`, not as zero. For a tensor dimension of length $n_d$, lag component
$\delta_d$ has

$$
N_{\mathrm{candidate}}(\boldsymbol{\delta})
=\prod_d \max(n_d-|\delta_d|,0)
$$

candidate pairs before masking. Thus a latitude lag of four cells is undefined
for a grid containing exactly four latitude cells, whereas a longitude lag of
four remains possible when the longitude dimension contains at least five cells.

## Direction-balanced lag aggregation

`aggregate_lag_information` groups sign-equivalent directions into shells that
keep horizontal, depth, and time grid distances separate. A family is the exact
active-role signature (horizontal, depth, temporal, or a particular mixture),
so depth-time and horizontal-time coupling are not merged in a 4-D tensor. For
metric $m$, directions are averaged equally within shell $s$ and shells are
averaged equally within family $f$:

$$
A_{s,m}=\frac{1}{|L_s|}\sum_{\ell\in L_s}m_\ell,
\qquad
A_{f,m}=\frac{1}{|S_f|}\sum_{s\in S_f}A_{s,m}.
$$

By default, a shell is used only when every requested direction in that shell is
applicable. Thus a grid with `lon=4` support but no `lat=4` support reports both
directions as excluded from its balanced aggregate; the available `lon=4` value
remains visible in the direction-resolved plot.

The three reported bounded lag metrics are dependence
$D_\ell=I_\ell/\max(H_{\ell,L},H_{\ell,R})$, coherence
$C_\ell=1-V_\ell$, and direction-balanced residual-coding savings. Exact pooled
coding savings are also reported from bit totals:

$$
S_{\mathrm{code}}=1-\frac{\sum_\ell E_\ell}{\sum_\ell F_\ell},
\qquad
b_{\mathrm{residual}}=\frac{\sum_\ell E_\ell}{\sum_\ell n_\ell}.
$$

Because the lag residual streams overlap, these pooled values compare the
hypothetical lag predictors on their combined residual samples; they are not an
estimate of the size of a single encoded tensor file.

For comparisons across binning schemes, `compare_lag_information` uses common
complete shells by default. If one direction in a shell is unavailable in any
scheme, the entire shell is removed from every scheme. This prevents a lone
longitude direction from representing a nominally direction-balanced spatial
distance.

```julia
comparison::NamedTuple = NMFk.compare_lag_information(structure_results)
heatmap::Gadfly.Plot = NMFk.plot_lag_information_aggregate_heatmap(
    structure_matrix,
    spatial_labels,
    temporal_labels,
)
```

These aggregates describe cell-relative regularity and compressibility. They
are not fractions of raw information preserved: one cell represents different
physical distances in different grids, sparse zero-filled tensors can appear
highly coherent, and independently fitted quantizers need not share thresholds.
The primary event-merging preservation measure remains

$$
R_{\mathrm{event}}=\frac{I(X;G)}{H(X)}
=1-\frac{H(X\mid G)}{H(X)}.
$$

`plot_information_retention_tradeoff` therefore plots $R_{\mathrm{event}}$
against the declared possible grid-cell count and uses aggregate lag dependence
only as point color. Pareto labels identify schemes for which retention cannot
be improved without increasing tensor size.

The raw/grid comparisons supplied to this plot must have been computed with an
explicit `grid_cell_count`; an occupied-cell count inferred from the observations
is not a dense storage or computation cost and is rejected.

The default `lag_sign=:canonical` treats opposite spatial offsets as the same
unordered pair and makes nonzero temporal lags point forward. It still keeps
opposite forward-time propagation directions such as `(1,1,1)` and
`(-1,-1,1)` separate. Use
`lag_sign=:directed` when forward and reverse conditional entropies must be
reported as distinct requested directions.

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

### Magnitude-mark aggregation

Three related questions should remain separate:

1. **Event-merging loss** is the localization uncertainty $H(X\mid G)$ described
   above. It asks which raw event state produced an event assigned to grid cell
   $G$.
2. **Count or occurrence structure** is measured from the gridded count tensor,
   including its multidimensional lag dependence and residual coding.
3. **Magnitude aggregation loss** concerns variation among event marks inside a
   cell. It requires an explicit magnitude quantizer and reconstruction model.

For a fixed magnitude precision $\Delta_M>0$ and origin $M_0$, define the
origin-anchored magnitude symbol

$$
Q_i=\left\lfloor\frac{M_i-M_0}{\Delta_M}\right\rfloor.
$$

The empirical within-cell magnitude heterogeneity is

$$
H(Q\mid G)
=\sum_g\frac{n_g}{N}H(Q\mid G=g)
=H(Q,G)-H(G).
$$

The implementation uses the event-weighted sparse-cell sum as the authoritative
value. The joint-entropy subtraction is retained as a scale-tolerant diagnostic,
because subtracting two large empirical entropies can lose floating-point
precision on high-cardinality grids.

`magnitude_aggregation_information` returns this quantity as
`magnitude_conditional_entropy_bits` and retains the nonzero per-cell symbol
counts in compressed-sparse-row form. It also reports

$$
I(Q;G)=H(Q)-H(Q\mid G),
$$

together with the normalized heterogeneity $H(Q\mid G)/H(Q)$ and magnitude
location-dependence fraction $I(Q;G)/H(Q)$. The latter is returned as
`magnitude_location_dependence_fraction`. The older
`magnitude_retention_fraction` field is an exact compatibility alias; it does
not measure retention by a maximum, mean, or median cell reconstruction. Both
fractions are undefined and returned as `NaN` when $H(Q)=0$. Here
$H(Q\mid G)$ is a quantized mark-heterogeneity measure and ideal conditional
mark-coding bound. It should not be interpreted as all physical information
lost from an earthquake catalog.

`magnitude_aggregation_loss_fraction` is an alias of the normalized quantized
heterogeneity $H(Q\mid G)/H(Q)$. It is conditional quantized-mark uncertainty,
not physical distortion or total catalog loss, and it does not depend on whether
the later reconstruction diagnostics use the maximum, mean, or median.

These are empirical plug-in estimates. Sparse, singleton-heavy grids can make
the apparent location dependence optimistic because a singleton grid label
memorizes its one observed mark. Report the returned singleton cell and event
fractions beside this statistic, and compare resolutions on a common event
cohort.

Magnitude heterogeneity is monotone under coarsening only when the grids form
genuinely nested deterministic partitions. Arbitrary bin-count sequences can
shift cell boundaries and need not be nested, so small nonmonotone changes across
such a sweep are not automatically implementation errors.

For a cell estimator $s\in\{\text{maximum},\text{mean},\text{median}\}$, let
$\widehat M_g^{(s)}$ be its physical reconstruction. The reported physical
errors are

$$
e_i^{(s)}=M_i-\widehat M_{G_i}^{(s)},
$$

$$
\mathrm{MAE}_s=\frac1N\sum_i\lvert e_i^{(s)}\rvert,
\qquad
\mathrm{RMSE}_s=\sqrt{\frac1N\sum_i\left(e_i^{(s)}\right)^2},
\qquad
\mathrm{bias}_s=\frac1N\sum_i e_i^{(s)}.
$$

The mean minimizes within-cell squared error, the median minimizes within-cell
absolute error, and the maximum preserves the largest observed cell mark rather
than minimizing either error. Earthquake magnitude is normally logarithmic, so
these errors remain in supplied magnitude units and are not linear energy-error
measures.

Coding uses a distinct, exactly invertible integer residual. The physical
predictor is mapped to its nearest lattice coordinate $\widehat Q_g^{(s)}$
(ties to even), and

$$
R_i^{(s)}=Q_i-\widehat Q_{G_i}^{(s)},
\qquad
Q_i=\widehat Q_{G_i}^{(s)}+R_i^{(s)}.
$$

Consequently, subtracting a cell-specific predictor is a conditional shift and

$$
H(R^{(s)}\mid G)=H(Q\mid G).
$$

The implementation verifies this identity. Pooled residual entropy
$H(R^{(s)})$ can be larger because residual distributions from different cells
are mixed. Both ideal Shannon bits and binary-Huffman bits are reported. The
common fixed-width reference uses the bit width of the global observed
magnitude-symbol span, making residual coding savings comparable across grids
and estimators. A second diagnostic uses the observed pooled residual range.

All coding sizes exclude predictor tables, grid labels, codebooks, headers, and
container overhead. In particular, a singleton cell can have zero conditional
residual payload only because its mark has moved into the uncounted cell
predictor. Singleton cell and observation fractions are therefore reported
explicitly.

The stored sparse histograms preserve the quantized magnitude multiset in each
cell, but not its pairing with event identity, subcell position, time, or input
order. Reconstructing an ordered catalog losslessly would additionally require
the event-aligned residual stream, every event's grid assignment, the predictor
table, and the coding metadata; those payloads are intentionally not retained by
this summary metric.

```julia
import NMFk

magnitude_information::NamedTuple = NMFk.magnitude_aggregation_information(
    magnitudes,
    (
        latitude=latitude_indices,
        longitude=longitude_indices,
        time=time_indices,
    );
    magnitude_precision=0.1,
    magnitude_origin=0.0,
    reconstructions=(:maximum, :mean, :median),
    residual_coding=:huffman,
)
```

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
    annotation=:comparison,
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
and red means severe merging. The compact default `annotation=:percent` prints
the empirical raw-baseline merging-loss percentage. The more explicit
`annotation=:comparison` prints three quantities in every cell: the percentage
lost, `H(X | G)` bits out of the common raw `H(X)` baseline, and the effective
ambiguity `2^H(X | G)`. This is often the clearest single figure for comparing a
two-dimensional family of grids with the raw observations.

## Oklahoma result persistence and reuse

The Oklahoma structure-analysis workflow writes a version-4 JLD2 checkpoint
before tensor work and after every completed grid configuration. The top-level
`analysis.rawdata_information` stores the raw baseline, precision/origin metadata,
state fingerprint, and feature residual summaries. A separate exact fingerprint
of all selected dates, coordinates, magnitudes, and their ordering prevents a
checkpoint from being resumed against changed measurement values. Every grid
result stores `count_mass`, `rawdata_comparison`, and `grid_shape` alongside the
existing maximum-magnitude and structure summaries. Version 4 also stores
`energy_rawdata_information`, the complete energy-proxy relation metadata, and
per-grid `energy_mass` and `energy_rawdata_comparison` results without storing a
dense energy tensor. Its `magnitude_aggregation` result retains compressed sparse
per-cell magnitude histograms and max/mean/median reconstruction summaries;
top-level configuration and scope metadata record exactly what is and is not
retained.

After including `oklahoma_structure_information_common.jl`, the persisted
raw/grid values can be inspected directly without rebuilding any tensor:

```julia
analysis::NamedTuple = load_oklahoma_information_results(results_path)
raw_information::NamedTuple = analysis.rawdata_information
time_label::String = first(keys(analysis.spatial_results))
resolution_index::Int = 1
comparison::NamedTuple =
    analysis.spatial_results[time_label][resolution_index].rawdata_comparison
magnitude_aggregation::NamedTuple =
    analysis.spatial_results[time_label][resolution_index].magnitude_aggregation

raw_entropy_bits::Float64 = comparison.raw_entropy_bits              # H(X)
grid_entropy_bits::Float64 = comparison.grid_entropy_bits            # H(G)
retained_bits::Float64 = comparison.retained_distinguishability_bits # I(X; G)
merging_loss_bits::Float64 = comparison.merging_loss_bits            # H(X | G)
magnitude_heterogeneity_bits::Float64 =
    magnitude_aggregation.magnitude_conditional_entropy_bits          # H(M | G)
```

`analysis.rawdata_information` contains the common raw-state definition and
baseline metadata. Each `rawdata_comparison` contains the paired empirical
comparison for one grid configuration.

The Oklahoma retained/lost baseline deliberately uses longitude, latitude, and
time. It therefore measures event-localization information lost by spatial and
temporal binning. Magnitude is analyzed separately: the magnitude-heterogeneity
heatmap displays empirical $H(M\mid G)/H(M)$, while companion figures compare
max/mean/median physical reconstruction errors and residual coding. This avoids
mislabeling entropy of the nonadditive maximum-magnitude tensor as raw
information retained.

### Additive event-energy proxy

When only earthquake magnitudes are available, an additive energy-like mark can
be defined explicitly as a proxy:

$$
E_i^{\rm proxy}=10^{aM_i+b},
\qquad
E_g^{\rm proxy}=\sum_{i:G_i=g}E_i^{\rm proxy}.
$$

The sum is performed in linear energy space. Magnitudes and logarithmic energies
must not be added. If the magnitude scale is unknown or mixed, the result is
called an energy proxy rather than physical radiated energy.

The Oklahoma workflow therefore defaults to $a=1.5$, $b=0$, and a unitless
relative-energy label. Two alternative calibrated configurations are provided.
The historical `GUTENBERG_RICHTER_ENERGY_CONFIGURATION` uses
$a=1.5$, $b=4.8$ in joules and is labeled as a generic energy-magnitude
relation rather than as moment-magnitude-specific. The
`USGS_MOMENT_MAGNITUDE_ENERGY_CONFIGURATION` preset uses $a=1.44$, $b=5.24$
in joules and should be selected only after confirming homogeneous moment
magnitudes. These relations are documented by the USGS in its
[historical generic magnitude-energy table](https://pubs.usgs.gov/of/1998/ofr-98-0767/)
and its current
[moment-magnitude energy overview](https://www.usgs.gov/programs/earthquake-hazards/earthquake-magnitude-energy-release-and-shaking-intensity).

Because entropy uses normalized weights, a constant intercept $b$ and the
reported energy unit do not change the weighted localization fractions. They do
change the stored total-energy scale. The coefficient $a$ changes relative event
weights and therefore can change the information results.

To ask where energetic events become indistinguishable after gridding, define

$$
q_i=\frac{E_i^{\rm proxy}}{\sum_jE_j^{\rm proxy}},
\qquad
q_x=\sum_{i:X_i=x}q_i,
\qquad
q_g=\sum_{i:G_i=g}q_i.
$$

The energy-weighted raw and gridded localization entropies are

$$
H_E(X)=-\sum_x q_x\log_2 q_x,
\qquad
H_E(G)=-\sum_g q_g\log_2 q_g,
$$

and, for deterministic grid assignment,

$$
H_E(X\mid G)=H_E(X)-H_E(G).
$$

Gridding conserves $\sum_g E_g^{\rm proxy}$; $H_E(X\mid G)$ measures the lost
ability to localize that energy among raw event states. This is complementary to
the unweighted event-localization loss, which gives every event equal weight.
When every event has a unique raw state, $q_x=q_i$ and the raw-state entropy
reduces to the event-index form $-\sum_iq_i\log_2q_i$.
The per-feature raw sequence residual summaries remain explicitly labeled
`weighting=:unweighted_sequence`; the energy weights apply to state/grid
localization entropy, not to those transition-residual histograms.

Consequently, `energy_mass` and `energy_rawdata_comparison` answer where and
when the energy proxy is localized, and how much of that localization is merged
by gridding. They are not entropy of energy-amplitude symbols, do not quantify
the within-cell distribution of event magnitudes, and do not yet describe lagged
structure of a summed-energy tensor. A few large events can dominate this
weighted view, so it is shown beside, rather than instead of, the unweighted
event-localization result.

### Oklahoma multidimensional lag analysis

The Oklahoma workflow keeps the plotted coordinates in longitude/latitude and
stores each spatial lag component in degrees, using the actual cell widths
returned by `NMFk.indicize`. It does not collapse longitude, latitude, and time
into one mixed-unit distance. Enabled lag runs use zero-preserving count
quantization, so an empty cell cannot share a symbol with a low positive count.
Legacy runs with `lag_offsets=nothing` retain linear quantization and their
existing cache behavior. Enable the recommended axial, diagonal, temporal, and
mixed space-time directions with:

```julia
analysis::NamedTuple = run_oklahoma_structure_analysis(
    dates,
    longitudes,
    latitudes,
    magnitudes,
    selection_mask,
    longitude_steps,
    latitude_steps,
    date_steps,
    DATASET_LABEL,
    OUTPUT_DIRECTORY;
    lag_offsets=RECOMMENDED_OKLAHOMA_LAG_OFFSETS,
)
```

Custom offsets can retain the intuitive public coordinate order:

```julia
custom_lags::Vector{NTuple{3,Int}} = [
    oklahoma_lag_offset(longitude=1),
    oklahoma_lag_offset(latitude=1, longitude=1),
    oklahoma_lag_offset(latitude=1, longitude=-1),
    oklahoma_lag_offset(latitude=1, longitude=1, time=1),
]
```

The helper converts those named components to the tensor's internal
`(latitude, longitude, time)` order. The default `lag_sign=:canonical` removes
only reverse duplicates while preserving distinct forward-time propagation
directions. Because a positive time component fixes the arrow of time, the
recommended set includes positive and negative axial propagation and all four
signed spatial diagonals at a one-bin temporal lag. With the default result
filename, an enabled run is automatically
stored as `<dataset>_lag_information_theory_results.jld2`. Its exact requested
offsets, quantization, and coordinate metadata are validated before a version-3
or version-4 checkpoint is reused;
older checkpoints are never silently treated as enabled lag analyses.

These Oklahoma lag metrics currently analyze the complete zero-filled event-count
field. Thus quiet-background pairs, including zero-zero pairs, are part of the
reported dependence and coherence. This is useful for occupancy persistence but
is distinct from an event-conditioned marked-point analysis.

`run_oklahoma_structure_analysis(...; resume=true)` is the default. It first
checks the deterministic result path for a version-4 JLD2 analysis and validates
the selected-input fingerprint and complete configuration. A complete matching
analysis is returned before raw metrics, grid indices, or tensor metrics are
recomputed. A partial matching analysis resumes at its first missing grid
configuration, and each computed grid is reused for both spatial and temporal
figure families. Set `save_figures=false` when only the precomputed `NamedTuple`
is needed; the default remains `true` so normal script runs still create or
refresh the figures.

A matching version-3 checkpoint is backed up with a `.v3.backup` suffix and
upgraded by adding the sparse magnitude comparisons. A matching version-2
checkpoint is backed up with a `.v2.backup` suffix and upgraded by adding both
energy-weighted and magnitude comparisons. These upgrades reuse the stored
tensor metrics and do not rebuild the dense count or maximum-magnitude tensors.

Version-1 checkpoints lack a complete historical input fingerprint and therefore
are not trusted automatically. If, and only if, the current inputs are exactly
those used for that checkpoint, pass `trust_version1_checkpoint=true`; the file
is then backed up with a `.v1.backup` suffix and upgraded by adding sparse
raw/grid, energy-weighted, and magnitude comparisons without recomputing its
expensive tensor metrics.

The recreated figure set includes
`<dataset>_rawdata_information_loss_heatmap.png`, which places spatial resolution
on one axis, temporal resolution on the other, and directly displays the
percentage of observed raw-state distinguishability lost through grid collisions
for every configuration. It also includes
`<dataset>_raw_vs_gridded_information.png`, whose detailed cell annotations make
the common raw baseline, loss bits, and effective ambiguity explicit.
The analogous
`<dataset>_energy_weighted_information_loss_heatmap.png` and
`<dataset>_energy_weighted_raw_vs_gridded_information.png` files answer the same
localization question after weighting every raw event by its additive energy
proxy. The eye-catching
`<dataset>_magnitude_heterogeneity_heatmap.png` adds normalized quantized
magnitude heterogeneity, $H(M\mid G)$ bits, singleton-event fraction, max versus
mean RMSE, and median MAE for every spatial/temporal configuration. Sweep figures
also compare the raw $H(M)$ baseline, ideal conditional $H(M\mid G)$ baseline,
pooled residual coding, physical reconstruction errors, and coding savings.

### Selecting a binning scheme

There is no data-independent single optimum: the finest candidate normally
retains the most empirical information, while the coarsest candidate costs the
least. `optimize_binning_information` therefore exposes the trade-off and keeps
the selection policy explicit.

For scheme $s$, define its possible-grid cost as

$$
C_s=\prod_{d=1}^{D} n_{s,d},
$$

where $n_{s,d}$ is the number of bins on dimension $d$. Let $R_{s,k}\in[0,1]$
be an information-retention metric, such as event, energy-weighted, longitude,
latitude, time, or magnitude retention. The relative retention within the
supplied candidate set has an explicit all-zero case:

$$
\widetilde R_{s,k}=
\begin{cases}
\dfrac{R_{s,k}}{\max_j R_{j,k}}, & \max_j R_{j,k}>0,\\
1, & \max_j R_{j,k}=0.
\end{cases}
$$

If a metric is zero for every candidate, the implementation records it in
`degenerate_metrics` and treats it as a neutral value of one in the balanced
summary.

A zero-entropy raw baseline is different: its retention fraction is undefined,
not 100%. Omit that metric from the optimization (or choose a scientifically
meaningful raw precision that yields nonzero entropy). The generic optimizer
rejects `NaN` and other nonfinite retention values rather than coercing them to
one.

The balanced retention is the worst retained metric,

$$
B_s=\min_{k\in K}\widetilde R_{s,k},
$$

where $K$ is the explicitly selected set of constraint metrics. The plotted
bottleneck label identifies the metric attaining this minimum. Relative values
measure performance against the best *observed candidate*; they are not a claim
of perfect retention.

A scheme is on the balanced cost-retention Pareto frontier when no other scheme
has both $C_j\le C_s$ and $B_j\ge B_s$, with at least one strict inequality. The
full multiobjective frontier applies the same definition to every
$\widetilde R_{s,k}$ separately.

The most interpretable unique selector is an epsilon constraint. Given declared
absolute minimum retentions $\rho_k$,

$$
C_{s^*}=\min_{s:\ R_{s,k}\ge\rho_k\ \text{for every constrained }k} C_s.
$$

The near-best policy is useful before defensible absolute thresholds are known.
For a requested fraction $q$ it selects

$$
C_{s_q^*}=\min_{s:\ \widetilde R_{s,k}\ge q\ \text{for every }k\in K} C_s.
$$

Thus the 90%, 95%, and 99% recommendations mean the least expensive supplied
schemes reaching that fraction of every metric's best observed value. They do
not mean 90%, 95%, or 99% of a theoretical continuum-limit truth.

Absolute `retention_targets` are a separate primary-metric-only policy. Each
target recommendation records `metric` and
`criterion=:minimum_cost_meeting_primary_metric_target` so it cannot be
mistaken for a constraint across every metric. Under a declared possible-cell
budget, the budget policy instead maximizes $B_s$ over feasible schemes and
records `criterion=:maximum_balanced_relative_retention`. Ties prefer the less
expensive scheme and then the earlier supplied candidate.

The exploratory knee normalizes $\log_{10}C_s$ and $B_s$ to $x_s,y_s\in[0,1]$
on the balanced Pareto frontier, then selects the interior point maximizing

$$
d_s=\frac{y_s-x_s}{\sqrt{2}}.
$$

It is reported as `applicable=false` for fewer than three usable Pareto points.
By default, the optimizer calls the elbow pronounced only when $d_s\ge0.05$; change
this explicit threshold with `knee_minimum_score`. A knee depends on the
supplied resolutions and must not be reported as a universal physical optimum.

```julia
import NMFk

grid_cell_counts::Vector{Int} = Int[1_000, 10_000, 100_000]
retention_metrics::NamedTuple = (
    event=Float64[0.70, 0.90, 0.96],
    energy=Float64[0.78, 0.93, 0.98],
)
labels::Vector{String} = String["coarse", "medium", "fine"]

optimization::NamedTuple = NMFk.optimize_binning_information(
    grid_cell_counts,
    retention_metrics,
    labels;
    primary_metric=:event,
    minimum_retentions=(event=0.90, energy=0.90),
    near_best_fractions=Float64[0.90, 0.95, 0.99],
    knee_minimum_score=0.05,
)
```

The raw-comparison overload records `rawdata_baseline=:states` or `:records`
and a corresponding `retention_semantics` field, so a saved optimization result
does not lose the meaning of its single `event` retention vector.

Directional lag metrics remain structure diagnostics rather than hidden terms
in this optimizer. A one-cell lag represents different physical distances and
durations in different grids, and coarse zero-filled tensors can appear more
coherent simply through smoothing. If lag structure is later made a selection
constraint, compare all candidates on common physical longitude/latitude/time
lag shells against a declared reference and verify that adding shells no longer
changes the ranking.

Finally, raw precision is part of the scientific definition of the baseline.
Very fine longitude, latitude, and time precision can make $H(X)$ nearly equal
to event-record identity. Optimization results should therefore store the raw
precision and origin, display them in the summary, and be repeated at realistic
measurement precisions and shifted grid origins before calling a scheme robust.

The Oklahoma dense sweep is implemented separately from the tensor/lag run:

```julia
include(joinpath(@__DIR__, "oklahoma_binning_optimization_common.jl"))

optimization_analysis::NamedTuple = run_oklahoma_binning_optimization(
    dates,
    longitudes,
    latitudes,
    magnitudes,
    selection_mask,
    dataset_label,
    output_directory,
)
```

It evaluates raw-to-grid comparisons without constructing dense tensors and
writes `<dataset>_binning_optimization_results.jld2`,
`<dataset>_binning_optimization.png`, and
`<dataset>_binning_optimization_summary.md`. The version-1 optimization cache is
independent of the version-4 structure/lag cache. Matching data, raw precision,
energy settings, and candidate grids are reused by default; changing only a
target, near-best fraction, or hard retention constraint recomputes the cheap
selection layer from the cached retention vectors.

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
