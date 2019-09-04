NMFk: Nonnegative Matrix Factorization using k-means clustering
================

<div style="text-align: left">
    <img src="logo/nmfk-logo.png" alt="nmfk" width=50%  max-width=250px;/>
</div>

### Installation

After starting Julia, execute:

```julia
import Pkg; Pkg.add("https://github.com/TensorDecompositions/NMFk.jl.git")
```

### Testing

```julia
Pkg.test("NMFk")
```

### Publications:

- Stanev, V., Vesselinov, V.V., Kusne, A.G., Antoszewski, G., Takeuchi,I., Alexandrov, B.A., Unsupervised Phase Mapping of X-ray Diffraction Data by Nonnegative Matrix Factorization Integrated with Custom Clustering, Nature Computational Materials, 10.1038/s41524-018-0099-2, 2018. [PDF](http://monty.gitlab.io/papers/Stanev%20et%20al%202018%20Unsupervised%20phase%20mapping%20of%20X-ray%20diffraction%20data%20by%20nonnegative%20matrix%20factorization%20integrated%20with%20custom%20clustering.pdf)
- Iliev, F.L., Stanev, V.G., Vesselinov, V.V., Alexandrov, B.S., Nonnegative Matrix Factorization for identification of unknown number of sources emitting delayed signals PLoS ONE, 10.1371/journal.pone.0193974. 2018. [PDF](http://monty.gitlab.io/papers/Iliev%20et%20al%202018%20Nonnegative%20Matrix%20Factorization%20for%20identification%20of%20unknown%20number%20of%20sources%20emitting%20delayed%20signals.pdf)
- Stanev, V.G., Iliev, F.L., Hansen, S.K., Vesselinov, V.V., Alexandrov, B.S., Identification of the release sources in advection-diffusion system by machine learning combined with Green function inverse method, Applied Mathematical Modelling, 10.1016/j.apm.2018.03.006, 2018. [PDF](http://monty.gitlab.io/papers/Stanev%20et%20al%202018%20Identification%20of%20release%20sources%20in%20advection-diffusion%20system%20by%20machine%20learning%20combined%20with%20Green's%20function%20inverse%20method.pdf)
- Vesselinov, V.V., O'Malley, D., Alexandrov, B.S., Contaminant source identification using semi-supervised machine learning, Journal of Contaminant Hydrology, 10.1016/j.jconhyd.2017.11.002, 2017. [PDF](http://monty.gitlab.io/papers/Vesselinov%202017%20Contaminant%20source%20identification%20using%20semi-supervised%20machine%20learning.pdf)
- Alexandrov, B., Vesselinov, V.V., Blind source separation for groundwater level analysis based on non-negative matrix factorization, Water Resources Research, 10.1002/2013WR015037, 2014. [PDF](http://monty.gitlab.io/papers/Alexandrov%20&%20Vesselinov%202014%20Blind%20source%20separation%20for%20groundwater%20pressure%20analysis%20based%20on%20nonnegative%20matrix%20factorization.pdf)

Research papers are also available at [Google Scholar](http://scholar.google.com/citations?user=sIFHVvwAAAAJ&hl=en), [ResearchGate](https://www.researchgate.net/profile/Velimir_Vesselinov) and [Academia.edu](https://lanl.academia.edu/monty)

### Presentations:

- O'Malley, D., Vesselinov, V.V., Alexandrov, B.S., Alexandrov, L.B., Nonnegative/binary matrix factorization with a D-Wave quantum annealer [PDF](http://monty.gitlab.io/presentations/OMalley%20et%20al%202017%20Nonnegative%20binary%20matrix%20factorization%20with%20a%20D-Wave%20quantum%20annealer.pdf)
- Vesselinov, V.V., Alexandrov, B.A, Model-free Source Identification, AGU Fall Meeting, San Francisco, CA, 2014. [PDF](http://monty.gitlab.io/presentations/vesselinov%20bss-agu2014-LA-UR-14-29163.pdf)

Presentations are also available at [slideshare.net](https://www.slideshare.net/VelimirmontyVesselin), [ResearchGate](https://www.researchgate.net/profile/Velimir_Vesselinov) and [Academia.edu](https://lanl.academia.edu/monty)

### Videos:

- Progress of nonnegative matrix factorization process:

<div style="text-align: left">
    <img src="movies/m643.gif" alt="nmfk-example" width=60%  max-width=250px;/>
</div>

Videos are also available at [YouTube](https://www.youtube.com/playlist?list=PLpVcrIWNlP22LfyIu5MSZ7WHp7q0MNjsj)

### Patent:

Alexandrov, B.S., Vesselinov, V.V., Alexandrov, L.B., Stanev, V., Iliev, F.L., Source identification by non-negative matrix factorization combined with semi-supervised clustering, [US20180060758A1](https://patents.google.com/patent/US20180060758A1/en)

For more information, visit [monty.gitlab.io](http://monty.gitlab.io)

### Examples:

* [Machine Learning](https://madsjulia.github.io/Mads.jl/Examples/machine_learning/index.html)
* [Blind Source Separation (i.e. Feature Extraction)](https://madsjulia.github.io/Mads.jl/Examples/blind_source_separation/index.html)
* [Source Identification](https://madsjulia.github.io/Mads.jl/Examples/contaminant_source_identification/index.html)

Installation behind a firewall
------------------------------

Julia uses git for package management. Add in the `.gitconfig` file in your home directory:

```
[url "https://"]
        insteadOf = git://
```

or execute:

```
git config --global url."https://".insteadOf git://
```

Set proxies:

```
export ftp_proxy=http://proxyout.<your_site>:8080
export rsync_proxy=http://proxyout.<your_site>:8080
export http_proxy=http://proxyout.<your_site>:8080
export https_proxy=http://proxyout.<your_site>:8080
export no_proxy=.<your_site>
```

For example, if you are doing this at LANL, you will need to execute the
following lines in your bash command-line environment:

```
export ftp_proxy=http://proxyout.lanl.gov:8080
export rsync_proxy=http://proxyout.lanl.gov:8080
export http_proxy=http://proxyout.lanl.gov:8080
export https_proxy=http://proxyout.lanl.gov:8080
export no_proxy=.lanl.gov
```

