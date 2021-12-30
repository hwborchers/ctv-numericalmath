## CRAN Task View: Numerical and Discrete Mathematics

|                 |                                                        |
|-----------------|--------------------------------------------------------|  
| **Maintainer:** | Hans W. Borchers                                       | 
| **Contact:**    | hwb at mailbox.org                                     | 
| **Version:**    | 2021-12-30                                             | 
| **URL:**        | <https://CRAN.R-project.org/view=NumericalMathematics> | 

This task view on numerical mathematics lists R packages and functions
that are useful for solving numerical problems in linear algebra and
analysis. It shows that R is a viable computing environment for
implementing and applying numerical methods, also outside the realm of
statistics.

The task view will *not* cover differential equations, optimization
problems and solvers, or packages and functions operating on times
series, because all these topics are treated extensively in the
corresponding task views
[DifferentialEquations](https://cran.r-project.org/web/views/DifferentialEquations.html),
[Optimization](https://cran.r-project.org/web/views/Optimization.html),
and [TimeSeries](https://cran.r-project.org/web/views/TimeSeries.html).
All these task views together will provide a good selection of what is
available in R for the area of numerical mathematics. The
[HighPerformanceComputing](https://cran.r-project.org/web/views/HighPerformanceComputing.html)
task view with its many links for parallel computing may also be of
interest.

The task view has been created to provide an overview of the topic. If
some packages are missing or certain topics in numerical math should be
treated in more detail, please let the maintainer know.

### Numerical Linear Algebra

As statistics is based to a large extent on linear algebra, many
numerical linear algebra routines are present in R, and some only
implicitly. Examples of explicitly available functions are vector and
matrix operations, matrix (QR) decompositions, solving linear equations,
eigenvalues/-vectors, singular value decomposition, or least-squares
approximation.

-   The recommended package [Matrix](https://cran.r-project.org/package=Matrix)
    provides classes and methods for dense and sparse matrices and
    operations on them, for example Cholesky and Schur decomposition,
    matrix exponential, or norms and conditional numbers for sparse
    matrices.
-   Recommended package [MASS](https://cran.r-project.org/package=MASS) adds
    generalized (Penrose) inverses and null spaces of matrices.
-   [expm](https://cran.r-project.org/package=expm) computes the exponential,
    logarithm, and square root of square matrices, but also powers of
    matrices or the Frechet derivative. `expm()` is to be preferred to
    the function with the same name in
    [Matrix](https://cran.r-project.org/package=Matrix).
-   [SparseM](https://cran.r-project.org/package=SparseM) provides classes and
    methods for sparse matrices and for solving linear and least-squares
    problems in sparse linear algebra
-   Package [rmumps](https://cran.r-project.org/package=rmumps) provides a wrapper
    for the MUMPS library, solving large linear systems of equations
    applying a parallel sparse direct solver
-   [sanic](https://cran.r-project.org/package=sanic) supports routines for solving
    (dense and sparse) large systems of linear equations; direct and
    iterative solvers from the Eigen C++ library are made available,
    including Cholesky, LU, QR, and Krylov subspace methods.
-   [Rlinsolve](https://cran.r-project.org/package=Rlinsolve) is a collection of
    iterative solvers for sparse linear system of equations; stationary
    iterative solvers such as Jacobi or Gauss-Seidel, as well as
    nonstationary (Krylov subspace) methods are provided.
-   [svd](https://cran.r-project.org/package=svd) provides R bindings to
    state-of-the-art implementations of singular value decomposition
    (SVD) and eigenvalue/eigenvector computations. Package
    [ssvd](https://cran.r-project.org/package=ssvd) will obtain sparse SVDs using an
    iterative thresholding method, while
    [irlba](https://cran.r-project.org/package=irlba) will compute approximate
    singular values/vectors of large matrices.
-   Package [PRIMME](https://cran.r-project.org/package=PRIMME) interfaces PRIMME, a
    C library for computing eigenvalues and eigenvectors of real
    symmetric or complex Hermitian matrices. It will find largest,
    smallest, or interior eigen-/singular values and will apply
    preconditioning to accelerate convergence.
-   The [geigen](https://cran.r-project.org/package=geigen) package computes
    generalized eigenvalues and -vectors for pairs of matrices and QZ
    (generalized Schur) decompositions.
-   [eigeninv](https://cran.r-project.org/package=eigeninv) generates matrices with
    a given set of eigenvalues ('inverse eigenvalue problem').
-   Package [rARPACK](https://cran.r-project.org/package=rARPACK), a wrapper for the
    ARPACK library, is typically used to compute only a few
    eigenvalues/vectors, e.g., a small number of largest eigenvalues.
-   Package [RSpectra](https://cran.r-project.org/package=RSpectra) interfaces the
    'Spectra' library for large-scale eigenvalue decomposition and SVD
    problems.
-   [optR](https://cran.r-project.org/package=optR) uses elementary methods of
    linear algebra (Gauss, LU, CGM, Cholesky) to solve linear systems.
-   Package [mbend](https://cran.r-project.org/package=mbend) for bending
    non-positive-definite (symmetric) matrices to positive-definiteness,
    using weighted and unweighted methods.
-   [matrixcalc](https://cran.r-project.org/package=matrixcalc) contains a
    collection of functions for matrix calculations, special matrices,
    and tests for matrix properties, e.g., (semi-)positive definiteness;
    mainly used for teaching and research purposes
-   [matlib](https://cran.r-project.org/package=matlib) contains a collection of
    matrix functions for teaching and learning matrix linear algebra as
    used in multivariate statistical methods; mainly for tutorial
    purposes in learning matrix algebra ideas using R.
-   Package [onion](https://cran.r-project.org/package=onion) contains routines for
    manipulating quaternions and octonians (normed division algebras
    over the real numbers); quaternions can be useful for handling
    rotations in three-dimensional space.
-   [clifford](https://cran.r-project.org/package=clifford) provides a suite of
    routines for arbitrary dimensional Clifford algebras and discusses
    special cases such as Lorentz transforms or quaternion
    multiplication.
-   Packages [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo) and
    [RcppEigen](https://cran.r-project.org/package=RcppEigen) enable the integration
    of the C++ template libraries 'Armadillo' resp. 'Eigen' for linear
    algebra applications written in C++ and integrated in R using
    [Rcpp](https://cran.r-project.org/package=Rcpp) for performance and ease of use.

### Special Functions

Many special mathematical functions are present in R, especially
logarithms and exponentials, trigonometric and hyperbolic functions, or
Bessel and Gamma functions. Many more special functions are available in
contributed packages.

-   Package [gsl](https://cran.r-project.org/package=gsl) provides an interface to
    the 'GNU Scientific Library' that contains implementations of many
    special functions, for example the Airy and Bessel functions,
    elliptic and exponential integrals, the hypergeometric function,
    Lambert's W function, and many more.
-   Airy and Bessel functions, for real and complex numbers, are also
    computed in package [Bessel](https://cran.r-project.org/package=Bessel), with
    approximations for large arguments.
-   Package [pracma](https://cran.r-project.org/package=pracma) includes special
    functions, such as error functions and inverses, incomplete and
    complex gamma function, exponential and logarithmic integrals,
    Fresnel integrals, the polygamma and the Dirichlet and Riemann zeta
    functions.
-   The hypergeometric (and generalized hypergeometric) function, is
    computed in [hypergeo](https://cran.r-project.org/package=hypergeo), including
    transformation formulas and special values of the parameters.
-   [HypergeoMat](https://cran.r-project.org/package=HypergeoMat) evaluates the
    hypergeometric functions of a matrix argument through a C++
    implementation of Koev and Edelman's algorithm.
-   Elliptic and modular functions are available in package
    [elliptic](https://cran.r-project.org/package=elliptic), including the
    Weierstrass P function and Jacobi's theta functions. There are tools
    for visualizing complex functions.
-   [Carlson](https://cran.r-project.org/package=Carlson) evaluates Carlson elliptic
    and incomplete elliptic integrals (with compex arguments).
-   Package [expint](https://cran.r-project.org/package=expint) wraps C-functions
    from the GNU Scientific Library to calculate exponential integrals
    and the incomplete Gamma function, including negative values for its
    first argument.
-   [fourierin](https://cran.r-project.org/package=fourierin) computes Fourier
    integrals of functions of one and two variables using the Fast
    Fourier Transform.
-   [logOfGamma](https://cran.r-project.org/package=logOfGamma) uses approximations
    to compute the natural logarithms of the Gamma function for large
    values.
-   Package [lamW](https://cran.r-project.org/package=lamW) implements both
    real-valued branches of the Lambert W function (using Rcpp).

### Polynomials

Function polyroot() in base R determines all zeros of a polynomial,
based on the Jenkins-Traub algorithm. Linear regression function lm()
can perform polynomial fitting when using `poly()` in the model formula
(with option `raw = TRUE`).

-   Packages [PolynomF](https://cran.r-project.org/package=PolynomF) (recommended)
    and [polynom](https://cran.r-project.org/package=polynom) provide similar
    functionality for manipulating univariate polynomials, like
    evaluating polynomials (Horner scheme), or finding their roots.
    'PolynomF' generates orthogonal polynomials and provides graphical
    display features.
-   [polyMatrix](https://cran.r-project.org/package=polyMatrix) (based on 'polynom')
    implements basic matrix operations and provides thus an
    infrastructure for the manipulation of polynomial matrices.
-   Package [MonoPoly](https://cran.r-project.org/package=MonoPoly) fits univariate
    polynomials to given data, applying different algorithms.
-   For multivariate polynomials, package
    [multipol](https://cran.r-project.org/package=multipol) provides various tools
    to manipulate and combine these polynomials of several variables.
-   Package [mpoly](https://cran.r-project.org/package=mpoly) facilitates symbolic
    manipulations on multivariate polynomials, including basic
    differential calculus operations on polynomials, plus some Groebner
    basis calculations.
-   [mvp](https://cran.r-project.org/package=mvp) enables fast manipulation of
    symbolic multivariate polynomials, using print and coercion methods
    from the 'mpoly' package, but offers speed improvements.
-   Package [orthopolynom](https://cran.r-project.org/package=orthopolynom) consists
    of a collection of functions to construct orthogonal polynomials and
    their recurrence relations, among them Chebyshev, Hermite, and
    Legendre polynomials, as well as spherical and ultraspherical
    polynomials. There are functions to operate on these polynomials.
-   Symbolic calculation and evaluation of the Jack polynomials, zonal
    polynomials (appear in random matrix theory), and Schur polynomials
    (appear in combinatorics) is available in package
    [jack](https://cran.r-project.org/package=jack).
-   The Free Algebra in R package
    [freealg](https://cran.r-project.org/package=freealg) handles multivariate
    polynomials with non-commuting indeterminates.

### Differentiation and Integration

`D()` and `deriv()` in base R compute derivatives of simple expressions
symbolically. Function `integrate()` implements an approach for
numerically integrating univariate functions in R. It applies adaptive
Gauss-Kronrod quadrature and can handle singularities and unbounded
domains to a certain extent.

-   Package [Deriv](https://cran.r-project.org/package=Deriv) provides an extended
    solution for symbolic differentiation in R; the user can add custom
    derivative rules, and the output for a function will be an
    executable function again.
-   [numDeriv](https://cran.r-project.org/package=numDeriv) sets the standard for
    numerical differentiation in R, providing numerical gradients,
    Jacobians, and Hessians, computed by simple finite differences,
    Richardson extrapolation, or the highly accurate complex step
    approach.
-   Package [dual](https://cran.r-project.org/package=dual) achieves automatic
    differentiation (for univariate functions) by employing dual
    numbers; for a mathematical function its value and its exact first
    derivative are returned.
-   Package [<span
    class="GitHub">autodiffr</span>](https://github.com/Non-Contradiction/autodiffr/)
    (on Github) provides an R wrapper for the Julia packages
    ForwardDiff.jl and ReverseDiff.jl to do automatic differentiation
    for native R functions.
-   [pracma](https://cran.r-project.org/package=pracma) contains functions for
    computing numerical derivatives, including Richardson extrapolation
    or complex step. `fderiv()` computes numerical derivatives of higher
    orders. [pracma](https://cran.r-project.org/package=pracma) has several routines
    for numerical integration: adaptive Lobatto quadrature, Romberg
    integration, Newton-Cotes formulas, Clenshaw-Curtis quadrature
    rules. `integral2()` integrates functions in two dimensions, also
    for domains characterized by polar coordinates or with variable
    interval limits.
-   Package [gaussquad](https://cran.r-project.org/package=gaussquad) contains a
    collection of functions to perform Gaussian quadrature, among them
    Chebyshev, Hermite, Laguerre, and Legendre quadrature rules,
    explicitly returning nodes and weights in each case. Function
    `gaussquad()` in package [statmod](https://cran.r-project.org/package=statmod)
    does a similar job.
-   [GramQuad](https://cran.r-project.org/package=GramQuad) allows for numerical
    integration based on Gram polynomials.
-   Package [fastGHQuad](https://cran.r-project.org/package=fastGHQuad) provides a
    fast [Rcpp](https://cran.r-project.org/package=Rcpp) -based implementation of
    (adaptive) Gauss-Hermite quadrature.
-   Adaptive multivariate integration over hyper-rectangles in
    n-dimensional space is available in package
    [cubature](https://cran.r-project.org/package=cubature) as function
    `adaptIntegrate()`, based on a C library of the same name. The
    integrand functions can even be multi-valued.
-   `vegas()` includes an approach to Monte Carlo integration based on
    importance sampling.
-   [mvQuad](https://cran.r-project.org/package=mvQuad) provides methods for
    generating multivariate grids that can be used for multivariate
    integration. These grids will be based on different quadrature rules
    such as Newton-Cotes or Gauss quadrature formulas.
-   Package [SparseGrid](https://cran.r-project.org/package=SparseGrid) provides
    another approach to multivariate integration in high-dimensional
    spaces. It creates sparse n-dimensional grids that can be used as
    with quadrature rules.
-   Package
    [SphericalCubature](https://cran.r-project.org/package=SphericalCubature)
    employs [cubature](https://cran.r-project.org/package=cubature) to integrate
    functions over unit spheres and balls in n-dimensional space;
    [SimplicialCubature](https://cran.r-project.org/package=SimplicialCubature)
    provides methods to integrate functions over m-dimensional simplices
    in n-dimensional space. Both packages comprise exact methods for
    polynomials.
-   Package [polyCub](https://cran.r-project.org/package=polyCub) holds some
    routines for numerical integration over polygonal domains in two
    dimensions.
-   Package [Pade](https://cran.r-project.org/package=Pade) calculates the numerator
    and denominator coefficients of the Pade approximation, given the
    Taylor series coefficients of sufficient length.
-   [calculus](https://cran.r-project.org/package=calculus) provides efficient
    functions for high-dimensional numerical and symbolic calculus,
    including accurate higher-order derivatives, Taylor series
    expansion, differential operators, and Monte-Carlo integration in
    orthogonal coordinate systems.
-   [features](https://cran.r-project.org/package=features) extracts features from
    functional data, such as first and second derivatives, or curvature
    at critical points, while
    [RootsExtremaInflections](https://cran.r-project.org/package=RootsExtremaInflections)
    finds roots, extrema and inflection points of curves defined by
    discrete points.

### Interpolation and Approximation

Base R provides functions `approx()` for constant and linear
interpolation, and `spline()` for cubic (Hermite) spline interpolation,
while `smooth.spline()` performs cubic spline approximation. Base
package splines creates periodic interpolation splines in function
`periodicSpline()`.

-   Interpolation of irregularly spaced data is possible with the
    [akima](https://cran.r-project.org/package=akima) package: `aspline()` for
    univariate data, `bicubic()` or `interp()` for data on a 2D
    rectangular domain. (This package is distributed under ACM license
    and not available for commercial use.)
-   Package [signal](https://cran.r-project.org/package=signal) contains several
    *filters* to smooth discrete data, notably `interp1()` for linear,
    spline, and cubic interpolation, `pchip()` for piecewise cubic
    Hermite interpolation, and `sgolay()` for Savitzky-Golay smoothing.
-   Package [pracma](https://cran.r-project.org/package=pracma) provides barycentric
    Lagrange interpolation (in 1 and 2 dimensions) in `barylag()` resp.
    `barylag2d()`, 1-dim. akima in `akimaInterp()`, and interpolation
    and approximation of data with rational functions, i.e. in the
    presence of singularities, in `ratinterp()` and `rationalfit()`.
-   The [interp](https://cran.r-project.org/package=interp) package provides
    bivariate data interpolation on regular and irregular grids, either
    linear or using splines. Currently the piecewise linear
    interpolation part is implemented. (It is intended to provide a free
    replacement for the ACM licensed `akima::interp` and
    `tripack::tri.mesh` functions.)
-   Package [chebpol](https://cran.r-project.org/package=chebpol) contains methods
    for creating multivariate Chebyshev and other multilinear
    interpolations on regular grids, e.g. the Floater-Hormann barycenter
    method, or polyharmonic splines for scattered data.
-   [tripack](https://cran.r-project.org/package=tripack) for triangulation of
    irregularly spaced data is a constrained two-dimensional Delaunay
    triangulation package providing both triangulation and generation of
    Voronoi mosaics of irregular spaced data.
-   `sinterp()` in package [stinepack](https://cran.r-project.org/package=stinepack)
    realizes interpolation based on piecewise rational functions by
    applying Stineman's algorithm. The interpolating function will be
    monotone in regions where the specified points change monotonically.
-   `Schumaker()` in package
    [schumaker](https://cran.r-project.org/package=schumaker) implements
    shape-preserving splines, guaranteed to be monotonic resp. concave
    or convex if the data is monotonic, concave, or convex.
-   [ADPF](https://cran.r-project.org/package=ADPF) uses least-squares polynomial
    regression and statistical testing to improve Savitzky-Golay
    smoothing.
-   Package [conicfit](https://cran.r-project.org/package=conicfit) provides several
    (geometric and algebraic) algorithms for fitting circles, ellipses,
    and conics in general.

### Root Finding and Fixed Points

`uniroot()`, implementing the Brent-Decker algorithm, is the basic
routine in R to find roots of univariate functions. There are
implementations of the bisection algorithm in several contributed
packages. For root finding with higher precision there is function
`unirootR()` in the multi-precision package
[Rmpfr](https://cran.r-project.org/package=Rmpfr). And for finding roots of
multivariate functions see the following packages:

-   Package [rootSolve](https://cran.r-project.org/package=rootSolve) includes
    function `multiroot()` for finding roots of systems of nonlinear
    (and linear) equations; it also contains an extension
    `uniroot.all()` that attempts to find all zeros of a univariate
    function in an intervall (excepting quadratic zeros).
-   For solving nonlinear systems of equations the
    [BB](https://cran.r-project.org/package=BB) package provides Barzilai-Borwein
    spectral methods in `sane()`, including a derivative-free variant in
    `dfsane()`, and multi-start features with sensitivity analysis.
-   Package [nleqslv](https://cran.r-project.org/package=nleqslv) solves nonlinear
    systems of equations using alternatively the Broyden or Newton
    method, supported by strategies such as line searches or trust
    regions.
-   [ktsolve](https://cran.r-project.org/package=ktsolve) defines a common interface
    for solving a set of equations with `BB` or `nleqslv`.
-   [FixedPoint](https://cran.r-project.org/package=FixedPoint) provides algorithms
    for finding fixed point vectors of functions, including Anderson
    acceleration, epsilon extrapolation methods, or minimal polynomial
    methods .
-   Package [daarem](https://cran.r-project.org/package=daarem) implements the
    DAAREM method for accelerating the convergence of any smooth,
    monotone, slow fixed point iteration.
-   Algorithms for accelerating the convergence of slow, monotone
    sequences from smooth contraction mappings such as the
    expectation-maximization (EM) algorithm are provided in packages
    [SQUAREM](https://cran.r-project.org/package=SQUAREM) resp.
    [turboEM](https://cran.r-project.org/package=turboEM).

### Discrete Mathematics and Number Theory

Not so many functions are available for computational number theory.
Note that integers in double precision can be represented exactly up to
`2^53 - 1`, above that limit a multi-precision package such as
[gmp](https://cran.r-project.org/package=gmp) is needed, see below.

-   Package [numbers](https://cran.r-project.org/package=numbers) provides functions
    for factorization, prime numbers, twin primes, primitive roots,
    modular inverses, extended GCD, etc. Included are some
    number-theoretic functions like divisor functions or Euler's Phi
    function.
-   [contfrac](https://cran.r-project.org/package=contfrac) contains various
    utilities for evaluating continued fractions and partial
    convergents.
-   [magic](https://cran.r-project.org/package=magic) creates and investigates
    magical squares and hypercubes, including functions for the
    manipulation and analysis of arbitrarily dimensioned arrays.
-   Package [freegroup](https://cran.r-project.org/package=freegroup) provides
    functionality for manipulating elements of a free group including
    juxtaposition, inversion, multiplication by a scalar, power
    operations, and Tietze forms.
-   The [partitions](https://cran.r-project.org/package=partitions) package
    enumerates additive partitions of integers, including restricted and
    unequal partitions.
-   [permutations](https://cran.r-project.org/package=permutations) treats
    permutations as invertible functions of finite sets and includes
    several mathematical operations on them.
-   Package [combinat](https://cran.r-project.org/package=combinat) generates all
    permutations or all combinations of a certain length of a set of
    elements (i.e. a vector); it also computes binomial coefficients.
-   Package [arrangements](https://cran.r-project.org/package=arrangements) provides
    generators and iterators for permutations, combinations and
    partitions. The iterators allow users to generate arrangements in a
    fast and memory efficient manner. Permutations and combinations can
    be drawn with/without replacement and support multisets.
-   Package [set6](https://cran.r-project.org/package=set6) implements (as R6
    classes) many forms of mathematical sets (sets, tuples, intervals)
    and allows for standard operations on them (unions, products,
    differences).
-   [RcppAlgos](https://cran.r-project.org/package=RcppAlgos) provides flexible
    functions for generating combinations or permutations of a vector
    with or without constraints; the extension package
    [RcppBigIntAlgos](https://cran.r-project.org/package=RcppBigIntAlgos) features a
    quadratic sieve algorithm for completely factoring large integers.
-   Package [Zseq](https://cran.r-project.org/package=Zseq) generates well-known
    integer sequences; the 'gmp' package is adopted for computing with
    arbitrarily large numbers. Every function has on its help page a
    hyperlink to the corresponding entry in the On-Line Encyclopedia of
    Integer Sequences ( [OEIS](https://oeis.org/)).
-   Package [primes](https://cran.r-project.org/package=primes) provides quite fast
    (Rcpp) functions for identifying and generating prime numbers. And
    [primefactr](https://cran.r-project.org/package=primefactr) uses prime
    factorization for computations such as reducing ratios of large
    factorials.

### Multi-Precision Arithmetic and Symbolic Mathematics

-   Multiple precision arithmetic is available in R through package
    [gmp](https://cran.r-project.org/package=gmp) that interfaces to the GMP C
    library. Examples are factorization of integers, a probabilistic
    prime number test, or operations on big rationals -- for which
    linear systems of equations can be solved.
-   Multiple precision floating point operations and functions are
    provided through package [Rmpfr](https://cran.r-project.org/package=Rmpfr) using
    the MPFR and GMP libraries. Special numbers and some special
    functions are included, as well as routines for root finding,
    integration, and optimization in arbitrary precision.
-   [Brobdingnag](https://cran.r-project.org/package=Brobdingnag) handles very large
    numbers by holding their logarithm plus a flag indicating their
    sign. (An excellent vignette explains how this is done using S4
    methods.)
-   [VeryLargeIntegers](https://cran.r-project.org/package=VeryLargeIntegers)
    implements a multi-precision library that allows to store and manage
    arbitrarily big integers; it includes probabilistic primality tests
    and factorization algorithms.
-   [bignum](https://cran.r-project.org/package=bignum) is a package for
    arbitrary-precision integer and floating-point numbers of 50 decimal
    digits of precision. The package utilizes the 'Boost.Multiprecision'
    C++ library and is specifically designed to work with the
    'tidyverse' collection of R packages.
-   Package [Ryacas](https://cran.r-project.org/package=Ryacas) interfaces the
    computer algebra system 'Yacas'; it supports symbolic and arbitrary
    precision computations in calculus and linear algebra.
-   Package [caracas](https://cran.r-project.org/package=caracas) (based on
    'reticulate') accesses the symbolic algebra system 'SymPy';
    supported are symbolic operations in linear algebra and calculus,
    such as eigenvalues, derivatives, integrals, limits, etc., computing
    special functions, or solving systems of equations.
-   Package [symengine](https://cran.r-project.org/package=symengine) provides an
    interface to 'SymEngine', a C++ library for fast symbolic
    calculations, such as manipulating mathematical expressions, finding
    exact derivatives, performing symbolic matrix computations, or
    solving ordinary differential equations (numerically).
-   Package [rim](https://cran.r-project.org/package=rim) provides an interface
    to the free and powerful computer algebra system 'Maxima'. Results can be
    output in 'LaTeX' or 'MathML' and 2D and 3D plots will be displayed
    directly. 'Maxima' code chunks can be included in 'RMarkdown' documents.
-   Package [m2r](https://cran.r-project.org/package=m2r) provides a persistent
    interface to Macauley2, an extended software program supporting research
    in algebraic geometry and commutative algebra. Macauley2 has to be
    installed independently, otherwise a Macauley2 process in the cloud will
    be instantiated.


### Python Interfaces

Python, through its modules 'NumPy', 'SciPy', 'Matplotlib', 'SymPy', and
'pandas', has elaborate and efficient numerical and graphical tools
available.

-   [reticulate](https://cran.r-project.org/package=reticulate) is an R interface to
    Python modules, classes, and functions. When calling Python in R
    data types are automatically converted to their equivalent Python
    types; when values are returned from Python to R they are converted
    back to R types. This package from the RStudio team is a kind of
    standard for calling Python from R.
-   [feather](https://cran.r-project.org/package=feather) provides bindings to read
    and write feather files, a lightweight binary data store designed
    for maximum speed. This storage format can also be accessed in
    Python, Julia, or Scala.
-   'pyRserve' is a Python module for connecting Python to an R process
    running [Rserve](https://cran.r-project.org/package=Rserve) as an RPC gateway.
    This R process can run on a remote machine, variable access and
    function calls will be delegated through the network.
-   [XRPython](https://cran.r-project.org/package=XRPython) (and 'XRJulia') are
    based on John Chambers' [XR](https://cran.r-project.org/package=XR) package and
    his "Extending R" book and allow for a structured integration of R
    with Python resp. Julia.

[SageMath](http://www.sagemath.org/) is an open source mathematics
system based on Python, allowing to run R functions, but also providing
access to systems like Maxima, GAP, FLINT, and many more math programs.
SageMath can be freely used through a Web interface at
[CoCalc](https://cocalc.com/).

### MATLAB, Octave, Julia, and other Interfaces

Interfaces to numerical computation software such as MATLAB (commercial)
or Octave (free) will be important when solving difficult numerical
problems. Unfortunately, at the moment there is no package allowing to
call Octave functions from within R.

-   The [matlab](https://cran.r-project.org/package=matlab) emulation package
    contains about 30 simple functions, replicating MATLAB functions,
    using the respective MATLAB names and being implemented in pure R.
-   Packages [rmatio](https://cran.r-project.org/package=rmatio) and
    [R.matlab](https://cran.r-project.org/package=R.matlab) provides tools to read
    and write MAT files (the MATLAB data format) for versions 4 and 5.
    'R.matlab' also enables a one-directional interface with a MATLAB v6
    process, sending and retrieving objects through a TCP connection.

Julia is "a high-level, high-performance dynamic programming language
for numerical computing", which makes it interesting for optimization
problems and other demanding scientific computations in R.

-   [JuliaCall](https://cran.r-project.org/package=JuliaCall) provides seamless
    integration between R and Julia; the user can call Julia functions
    just like any R function, and R functions can be called in the Julia
    environment, both with reasonable automatic type conversion. [Notes
    on Julia Call](https://hwborchers.github.io/) provides an
    introduction of how to apply Julia functions with 'JuliaCall'.
-   [JuliaConnectoR](https://cran.r-project.org/package=JuliaConnectoR) provides a
    functionally oriented interface for integrating Julia with R;
    imported Julia functions can be called as R functions; data
    structures are converted automatically.
-   Package [XRJulia](https://cran.r-project.org/package=XRJulia) provides an
    interface from R to computations in the Julia language, based on the
    interface structure described in the book "Extending R" by John M.
    Chambers.

Java Math functions can be employed through the 'rjava' or 'rscala' interfaces.
Then package [commonsMath](https://cran.r-project.org/package=commonsMath) allows
calling Java JAR files of the Apache Commons Mathematics Library, a specialized
library for all aspects of numerics, optimization, and differential equations.

Please note that commercial programs such as MATLAB, Maple, or Mathematica
have facilities to call R functions.

### CRAN packages:

  [ADPF](https://cran.r-project.org/package=ADPF)
  [akima](https://cran.r-project.org/package=akima)
  [arrangements](https://cran.r-project.org/package=arrangements)
  [BB](https://cran.r-project.org/package=BB)
  [Bessel](https://cran.r-project.org/package=Bessel)
  [bignum](https://cran.r-project.org/package=bignum)
  [Brobdingnag](https://cran.r-project.org/package=Brobdingnag)
  [calculus](https://cran.r-project.org/package=calculus)
  [caracas](https://cran.r-project.org/package=caracas)
  [Carlson](https://cran.r-project.org/package=Carlson)
  [chebpol](https://cran.r-project.org/package=chebpol)
  [clifford](https://cran.r-project.org/package=clifford)
  [combinat](https://cran.r-project.org/package=combinat)
  [commonsMath](https://cran.r-project.org/package=commonsMath)
  [conicfit](https://cran.r-project.org/package=conicfit)
  [contfrac](https://cran.r-project.org/package=contfrac)
  [cubature](https://cran.r-project.org/package=cubature)
  [daarem](https://cran.r-project.org/package=daarem)
  [Deriv](https://cran.r-project.org/package=Deriv) (core)
  [dual](https://cran.r-project.org/package=dual)
  [eigeninv](https://cran.r-project.org/package=eigeninv)
  [elliptic](https://cran.r-project.org/package=elliptic)
  [expint](https://cran.r-project.org/package=expint)
  [expm](https://cran.r-project.org/package=expm)
  [fastGHQuad](https://cran.r-project.org/package=fastGHQuad)
  [feather](https://cran.r-project.org/package=feather)
  [features](https://cran.r-project.org/package=features)
  [FixedPoint](https://cran.r-project.org/package=FixedPoint)
  [fourierin](https://cran.r-project.org/package=fourierin)
  [freealg](https://cran.r-project.org/package=freealg)
  [freegroup](https://cran.r-project.org/package=freegroup)
  [gaussquad](https://cran.r-project.org/package=gaussquad)
  [geigen](https://cran.r-project.org/package=geigen)
  [gmp](https://cran.r-project.org/package=gmp)
  [GramQuad](https://cran.r-project.org/package=GramQuad)
  [gsl](https://cran.r-project.org/package=gsl)
  [hypergeo](https://cran.r-project.org/package=hypergeo)
  [HypergeoMat](https://cran.r-project.org/package=HypergeoMat)
  [interp](https://cran.r-project.org/package=interp)
  [irlba](https://cran.r-project.org/package=irlba)
  [jack](https://cran.r-project.org/package=jack)
  [JuliaCall](https://cran.r-project.org/package=JuliaCall)
  [JuliaConnectoR](https://cran.r-project.org/package=JuliaConnectoR)
  [ktsolve](https://cran.r-project.org/package=ktsolve)
  [lamW](https://cran.r-project.org/package=lamW)
  [logOfGamma](https://cran.r-project.org/package=logOfGamma)
  [m2r](https://cran.r-project.org/package=m2r)
  [magic](https://cran.r-project.org/package=magic)
  [MASS](https://cran.r-project.org/package=MASS)
  [matlab](https://cran.r-project.org/package=matlab)
  [matlib](https://cran.r-project.org/package=matlib)
  [Matrix](https://cran.r-project.org/package=Matrix) (core)
  [matrixcalc](https://cran.r-project.org/package=matrixcalc)
  [mbend](https://cran.r-project.org/package=mbend)
  [MonoPoly](https://cran.r-project.org/package=MonoPoly)
  [mpoly](https://cran.r-project.org/package=mpoly)
  [multipol](https://cran.r-project.org/package=multipol)
  [mvp](https://cran.r-project.org/package=mvp)
  [mvQuad](https://cran.r-project.org/package=mvQuad)
  [nleqslv](https://cran.r-project.org/package=nleqslv)
  [numbers](https://cran.r-project.org/package=numbers)
  [numDeriv](https://cran.r-project.org/package=numDeriv) (core)
  [onion](https://cran.r-project.org/package=onion)
  [optR](https://cran.r-project.org/package=optR)
  [orthopolynom](https://cran.r-project.org/package=orthopolynom)
  [Pade](https://cran.r-project.org/package=Pade)
  [partitions](https://cran.r-project.org/package=partitions)
  [permutations](https://cran.r-project.org/package=permutations)
  [polyCub](https://cran.r-project.org/package=polyCub)
  [polyMatrix](https://cran.r-project.org/package=polyMatrix)
  [polynom](https://cran.r-project.org/package=polynom)
  [PolynomF](https://cran.r-project.org/package=PolynomF) (core)
  [pracma](https://cran.r-project.org/package=pracma) (core)
  [primefactr](https://cran.r-project.org/package=primefactr)
  [primes](https://cran.r-project.org/package=primes)
  [PRIMME](https://cran.r-project.org/package=PRIMME)
  [R.matlab](https://cran.r-project.org/package=R.matlab)
  [rARPACK](https://cran.r-project.org/package=rARPACK)
  [Rcpp](https://cran.r-project.org/package=Rcpp)
  [RcppAlgos](https://cran.r-project.org/package=RcppAlgos)
  [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo)
  [RcppBigIntAlgos](https://cran.r-project.org/package=RcppBigIntAlgos)
  [RcppEigen](https://cran.r-project.org/package=RcppEigen)
  [reticulate](https://cran.r-project.org/package=reticulate)
  [rim](https://cran.r-project.org/package=rim)
  [Rlinsolve](https://cran.r-project.org/package=Rlinsolve)
  [rmatio](https://cran.r-project.org/package=rmatio)
  [Rmpfr](https://cran.r-project.org/package=Rmpfr)
  [rmumps](https://cran.r-project.org/package=rmumps)
  [RootsExtremaInflections](https://cran.r-project.org/package=RootsExtremaInflections)
  [rootSolve](https://cran.r-project.org/package=rootSolve)
  [Rserve](https://cran.r-project.org/package=Rserve)
  [RSpectra](https://cran.r-project.org/package=RSpectra)
  [Ryacas](https://cran.r-project.org/package=Ryacas)
  [sanic](https://cran.r-project.org/package=sanic)
  [schumaker](https://cran.r-project.org/package=schumaker)
  [set6](https://cran.r-project.org/package=set6)
  [signal](https://cran.r-project.org/package=signal)
  [SimplicialCubature](https://cran.r-project.org/package=SimplicialCubature)
  [SparseGrid](https://cran.r-project.org/package=SparseGrid)
  [SparseM](https://cran.r-project.org/package=SparseM)
  [SphericalCubature](https://cran.r-project.org/package=SphericalCubature)
  [SQUAREM](https://cran.r-project.org/package=SQUAREM)
  [ssvd](https://cran.r-project.org/package=ssvd)
  [statmod](https://cran.r-project.org/package=statmod)
  [stinepack](https://cran.r-project.org/package=stinepack)
  [svd](https://cran.r-project.org/package=svd)
  [symengine](https://cran.r-project.org/package=symengine)
  [tripack](https://cran.r-project.org/package=tripack)
  [turboEM](https://cran.r-project.org/package=turboEM)
  [VeryLargeIntegers](https://cran.r-project.org/package=VeryLargeIntegers)
  [XR](https://cran.r-project.org/package=XR)
  [XRJulia](https://cran.r-project.org/package=XRJulia)
  [XRPython](https://cran.r-project.org/package=XRPython)
  [Zseq](https://cran.r-project.org/package=Zseq)

### Related links:

-   CRAN Task View: [DifferentialEquations](DifferentialEquations.html)
-   CRAN Task View: [Optimization](Optimization.html)
-   CRAN Task View: [TimeSeries](TimeSeries.html)
-   CRAN Task View:
    [HighPerformanceComputing](HighPerformanceComputing.html)
-   Textbook: [Hands-On Matrix Algebra Using
    R](http://www.worldscientific.com/worldscibooks/10.1142/7814)
-   Textbook: [Introduction to Scientific Programming and Simulation
    Using
    R](https://www.routledge.com/Introduction-to-Scientific-Programming-and-Simulation-Using-R/Jones-Maillardet-Robinson/p/book/9781466569997)
-   Textbook: [Numerical Methods in Science and Engineering Using
    R](https://www.routledge.com/Using-R-for-Numerical-Analysis-in-Science-and-Engineering/Bloomfield/p/book/9781439884485)
-   Textbook: [Computational Methods for Numerical Analysis with
    R](https://www.crcpress.com/Computational-Methods-for-Numerical-Analysis-with-R/II/p/book/9781498723633)
-   [MATLAB / R Reference (D.
    Hiebeler)](https://umaine.edu/mathematics/david-hiebeler/computing-software/matlab-r-reference/)
-   [Abramowitz and Stegun. Handbook of Mathematical
    Functions](http://people.math.sfu.ca/~cbm/aands/)
-   [Numerical Recipes: The Art of Numerical
    Computing](http://numerical.recipes/)
-   [E. Weisstein's Wolfram MathWorld](http://mathworld.wolfram.com/)
