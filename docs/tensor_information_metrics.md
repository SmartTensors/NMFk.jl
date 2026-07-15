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
\operatorname{clamp}\left(
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
b_{\mathrm{fixed}}
=
\left\lceil\log_2(2B-1)\right\rceil
$$

bits per residual, with total length

$$
L_{\mathrm{fixed}}
=
N_t\left\lceil\log_2(2B-1)\right\rceil.
$$

### Shannon limit

The ideal entropy-coding limit is

$$
b_{\mathrm{Shannon}}
=
H(R)
=
-\sum_r p_R(r)\log_2p_R(r).
$$

This is a theoretical lower bound rather than a concrete code-stream length.

### Huffman coding

If $n_r$ is the observed frequency of residual $r$ and $\ell_r$ is its binary
Huffman codeword length, then

$$
b_{\mathrm{Huffman}}
=
\frac{1}{N_t}\sum_r n_r\ell_r
=
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
& \max\!\left(H(X),H(Y)\right)>0,\\[8pt]
0, & \text{otherwise}.
\end{cases}
$$

Let $\mathcal S$ denote the set of axes classified as spatial. Reported spatial
dependence is

$$
D_{\mathrm{spatial}}
=
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
V_d
=
\frac{1}{N_d(B-1)}
\sum_{k=1}^{N_d}|Y_k-X_k|,
$$

where $N_d$ is the number of valid adjacent pairs along that axis.

Mean spatial variation is

$$
V_{\mathrm{spatial}}
=
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
D_{\mathrm{temporal}}
=
\begin{cases}
\dfrac{
H(Q_t)+H(Q_{t+1})-H(Q_t,Q_{t+1})
}{
\max\!\left(H(Q_t),H(Q_{t+1})\right)
},
& \max\!\left(H(Q_t),H(Q_{t+1})\right)>0,\\[10pt]
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
C_{\mathrm{temporal}}
=
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
L_{\mathrm{fixed}}
=
N_t\left\lceil\log_2(2B-1)\right\rceil
$$

be the fixed-width residual length, and let $L_{\mathrm{coded}}$ be the residual
length produced by the selected Shannon or Huffman model. Residual coding savings
is

$$
S_{\mathrm{coding}}
=
\operatorname{clamp}\left(
1-\frac{L_{\mathrm{coded}}}{L_{\mathrm{fixed}}},
0,1
\right).
$$

Before clamping, this is equivalently

$$
S_{\mathrm{coding}}
=
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
