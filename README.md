CRAN Task View: Numerical Mathematics
-------------------------------------

|                 |                                                        |
|-----------------|--------------------------------------------------------|  
| **Maintainer:** | Hans W. Borchers                                       | 
| **Contact:**    | hwb at mailbox.org                                     | 
| **Version:**    | 2018-08-12                                             | 
| **URL:**        | <https://CRAN.R-project.org/view=NumericalMathematics> | 

This task view on numerical mathematics lists R packages and functions that are useful for solving numerical problems in linear algebra and analysis. It shows that R is a viable computing environment for implementing and applying numerical methods, also outside the realm of statistics.

The task view will *not* cover differential equations, optimization problems and solvers, or packages and functions operating on times series, because all these topics are treated extensively in the corresponding task views [DifferentialEquations](https://cran.r-project.org/web/views/DifferentialEquations.html), [Optimization](https://cran.r-project.org/web/views/Optimization.html), and [TimeSeries](https://cran.r-project.org/web/views/TimeSeries.html). All these task views together will provide a good selection of what is available in R for the area of numerical mathematics. The [HighPerformanceComputing](https://cran.r-project.org/web/views/HighPerformanceComputing.html) task view with its many links for parallel computing may also be of interest.

The task view has been created to provide an overview of the topic. If some packages are missing or certain topics in numerical math should be treated in more detail, please let the maintainer know.

**Numerical Linear Algebra**

As statistics is based to a large extent on linear algebra, many numerical linear algebra routines are present in R, and some only implicitly. Examples of explicitly available functions are vector and matrix operations, matrix (QR) decompositions, solving linear equations, eigenvalues/-vectors, singular value decomposition, or least-squares approximation.

-   The recommended package [Matrix](https://cran.r-project.org/package=Matrix) provides classes and methods for dense and sparse matrices and operations on them, for example Cholesky and Schur decomposition, matrix exponential, or norms and conditional numbers for sparse matrices.
-   Recommended package [MASS](https://cran.r-project.org/package=MASS) adds generalized (Penrose) inverses and null spaces of matrices.
-   [expm](https://cran.r-project.org/package=expm/index.html) computes the exponential, logarithm, and square root of square matrices, but also powers of matrices or the Frechet derivative. `expm()` is to be preferred to the function with the same name in [Matrix](../packages/Matrix).
-   [SparseM](https://cran.r-project.org/package=SparseM) provides classes and methods for sparse matrices and for solving linear and least-squares problems in sparse linear algebra
-   Package [rmumps](https://cran.r-project.org/package=rmumps) provides a wrapper for the MUMPS library, solving large linear systems of equations applying a parallel sparse direct solver
-   [Rlinsolve](https://cran.r-project.org/package=Rlinsolve) is a collection of iterative solvers for sparse linear system of equations. Stationary iterative solvers such as Jacobi or Gauss-Seidel, as well as nonstationary (Krylov subspace) methods are provided.
-   [SolveLS](https://cran.r-project.org/package=SolveLS) provides basic stationary iterative solvers such as Jacobi, Gauss-Seidel, Successive Over-Relaxation and SSOR methods. Nonstationary, also known as Krylov subspace, methods are provided. Sparse matrix computation is supported through the 'Matrix' and 'RcppArmadillo' packages.
-   [svd](https://cran.r-project.org/package=svd/index.html) provides R bindings to state-of-the-art implementations of singular value decomposition (SVD) and eigenvalue/eigenvector computations. Package [ssvd](../packages/ssvd/index.html) will obtain sparse SVDs using an iterative thresholding method, while [irlba](../packages/irlba) will compute approximate singular values/vectors of large matrices.
-   Package [PRIMME](https://cran.r-project.org/package=PRIMME) interfaces PRIMME, a C library for computing eigenvalues and corresponding eigenvectors of real symmetric or complex Hermitian matrices. It can find largest, smallest, or interior eigen-/singular values and will apply preconditioning to accelerate convergence.
-   The packages [geigen](https://cran.r-project.org/package=geigen/index.html) and [QZ](../packages/QZ) compute generalized eigenvalues and -vectors for pairs of matrices, and QZ (generalized Schur) decompositions.
-   [eigeninv](https://cran.r-project.org/package=eigeninv) generates matrices with a given set of eigenvalues ('inverse eigenvalue problem').
-   Package [rARPACK](https://cran.r-project.org/package=rARPACK), a wrapper for the ARPACK library, is typically used to compute only a few eigenvalues/vectors, e.g., a small number of largest eigenvalues.
-   Package [RSpectra](https://cran.r-project.org/package=RSpectra) interfaces the 'Spectra' library for large-scale eigenvalue decomposition and SVD problems.
-   [optR](https://cran.r-project.org/package=optR) uses elementary methods of linear algebra (Gauss, LU, CGM, Cholesky) to solve linear systems.
-   [matrixcalc](https://cran.r-project.org/package=matrixcalc) contains a collection of functions for matrix calculations, special matrices, and tests for matrix properties, e.g., (semi-)positive definiteness.
-   Package [onion](https://cran.r-project.org/package=onion) contains routines for manipulating quaternions and octonians (normed division algebras over the real numbers); quaternions can be useful for handling rotations in three-dimensional space.
-   Packages [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo/index.html) and [RcppEigen](../packages/RcppEigen/index.html) enable the integration of the C++ template libraries 'Armadillo' resp. 'Eigen' for linear algebra applications written in C++ and integrated in R using [Rcpp](../packages/Rcpp) for performance and ease of use.

**Special Functions**

Many special mathematical functions are present in R, especially logarithms and exponentials, trigonometric and hyperbolic functions, or Bessel and Gamma functions. Many more special functions are available in contributed packages.

-   Package [gsl](https://cran.r-project.org/package=gsl) provides an interface to the 'GNU Scientific Library' that contains implementations of many special functions, for example the Airy and Bessel functions, elliptic and exponential integrals, the hypergeometric function, Lambert's W function, and many more.
-   Airy and Bessel functions, for real and complex numbers, are also computed in package [Bessel](https://cran.r-project.org/package=Bessel), with approximations for large arguments.
-   Package [pracma](https://cran.r-project.org/package=pracma) includes special functions, such as error functions and inverses, incomplete and complex gamma function, exponential and logarithmic integrals, Fresnel integrals, the polygamma and the Dirichlet and Riemann zeta functions.
-   [appell](https://cran.r-project.org/package=appell) computes Gauss' 2F1 and Appell's F1 hypergeometric functions for complex parameters and arguments quite accurately.
-   The hypergeometric (and generalized hypergeometric) function, is computed in [hypergeo](https://cran.r-project.org/package=hypergeo), including transformation formulas and special values of the parameters.
-   Elliptic and modular functions are available in package [elliptic](https://cran.r-project.org/package=elliptic), including the Weierstrass P function and Jacobi's theta functions. There are tools for visualizing complex functions.
-   Package [expint](https://cran.r-project.org/package=expint) wraps C-functions from the GNU Scientific Library to calculate exponential integrals and the incomplete Gamma function, including negative values for its first argument.
-   [fourierin](https://cran.r-project.org/package=fourierin) computes Fourier integrals of functions of one and two variables using the Fast Fourier Transform.
-   [logOfGamma](https://cran.r-project.org/package=logOfGamma) uses approximations to compute the natural logarithms of the Gamma function for large values.
-   Package [lamW](https://cran.r-project.org/package=lamW) implements both real-valued branches of the Lambert W function (using Rcpp).

**Polynomials**

Function polyroot() in base R determines all zeros of a polynomial, based on the Jenkins-Traub algorithm. Linear regression function lm() can perform polynomial fitting when using `poly()` in the model formula (with option `raw = TRUE`).

-   Packages [polynom](https://cran.r-project.org/package=polynom/index.html) and [PolynomF](../packages/PolynomF) provide similar functionality for manipulating univariate polynomials, like evaluating polynomials (Horner scheme), differentiating or integrating them, or solving polynomials, i.e. finding all roots (based on an eigenvalue computation).
-   Package [MonoPoly](https://cran.r-project.org/package=MonoPoly) fits univariate polynomials to given data, applying different algorithms.
-   For multivariate polynomials, package [multipol](https://cran.r-project.org/package=multipol) provides various tools to manipulate and combine these polynomials of several variables.
-   Package [mpoly](https://cran.r-project.org/package=mpoly) facilitates symbolic manipulations on multivariate polynomials, including basic differential calculus operations on polynomials, plus some Groebner basis calculations.
-   Package [orthopolynom](https://cran.r-project.org/package=orthopolynom) consists of a collection of functions to construct orthogonal polynomials and their recurrence relations, among them Chebyshev, Hermite, and Legendre polynomials, as well as spherical and ultraspherical polynomials. There are functions to operate on these polynomials.

**Differentiation and Integration**

`D()` and `deriv()` in base R compute derivatives of simple expressions symbolically. Function `integrate()` implements an approach for numerically integrating univariate functions in R. It applies adaptive Gauss-Kronrod quadrature and can handle singularities and unbounded domains to a certain extent.

-   Package [Deriv](https://cran.r-project.org/package=Deriv) provides an extended solution for symbolic differentiation in R; the user can add custom derivative rules, and the output for a function will be an executable function again.
-   [numDeriv](https://cran.r-project.org/package=numDeriv) sets the standard for numerical differentiation in R, providing numerical gradients, Jacobians, and Hessians, computed by simple finite differences, Richardson extrapolation, or the highly accurate complex step approach.
-   Package [<span class="GitHub">autodiffr</span>](https://github.com/Non-Contradiction/autodiffr/) (on Github) provides an R wrapper for the Julia packages ForwardDiff.jl and ReverseDiff.jl to do automatic differentiation for native R functions. (Works only with Julia v0.6 for the moment)
-   [pracma](https://cran.r-project.org/package=pracma/index.html) contains functions for computing numerical derivatives, including Richardson extrapolation or complex step. `fderiv()` computes numerical derivatives of higher orders. [pracma](../packages/pracma) has several routines for numerical integration: adaptive Lobatto quadrature, Romberg integration, Newton-Cotes formulas, Clenshaw-Curtis quadrature rules. `integral2()` integrates functions in two dimensions, also for domains characterized by polar coordinates or with variable interval limits.
-   Package [gaussquad](https://cran.r-project.org/package=gaussquad/index.html) contains a collection of functions to perform Gaussian quadrature, among them Chebyshev, Hermite, Laguerre, and Legendre quadrature rules, explicitly returning nodes and weights in each case. Function `gaussquad()` in package [statmod](../packages/statmod) does a similar job.
-   Package [fastGHQuad](https://cran.r-project.org/package=fastGHQuad/index.html) provides a fast [Rcpp](../packages/Rcpp) -based implementation of (adaptive) Gauss-Hermite quadrature.
-   Adaptive multivariate integration over hyper-rectangles in n-dimensional space is available in package [cubature](https://cran.r-project.org/package=cubature) as function `adaptIntegrate()`, based on a C library of the same name. The integrand functions can even be multi-valued.
-   Multi-dimensional numerical integration is also covered in package [R2Cuba](https://cran.r-project.org/package=R2Cuba), a wrapper around the C library Cuba. With `vegas()` it includes an approach to Monte Carlo integration based on importance sampling.
-   [mvQuad](https://cran.r-project.org/package=mvQuad) provides methods for generating multivariate grids that can be used for multivariate integration. These grids will be based on different quadrature rules such as Newton-Cotes or Gauss quadrature formulas.
-   Package [SparseGrid](https://cran.r-project.org/package=SparseGrid) provides another approach to multivariate integration in high-dimensional spaces. It creates sparse n-dimensional grids that can be used as with quadrature rules.
-   Package [SphericalCubature](https://cran.r-project.org/package=SphericalCubature/index.html) employs [cubature](../packages/cubature/index.html) to integrate functions over unit spheres and balls in n-dimensional space; [SimplicialCubature](../packages/SimplicialCubature) provides methods to integrate functions over m-dimensional simplices in n-dimensional space. Both packages comprise exact methods for polynomials.
-   Package [polyCub](https://cran.r-project.org/package=polyCub) holds some routines for numerical integration over polygonal domains in two dimensions.
-   Package [Pade](https://cran.r-project.org/package=Pade) calculates the numerator and denominator coefficients of the Pade approximation, given the Taylor series coefficients of sufficient length.
-   [features](https://cran.r-project.org/package=features/index.html) extracts features from functional data, such as first and second derivatives, or curvature at critical points, while [RootsExtremaInflections](../packages/RootsExtremaInflections) finds roots, extrema and inflection points of curves defined by discrete points.

**Interpolation and Approximation**

Base R provides functions `approx()` for constant and linear interpolation, and `spline()` for cubic (Hermite) spline interpolation, while `smooth.spline()` performs cubic spline approximation. Base package splines creates periodic interpolation splines in function `periodicSpline()`.

-   Interpolation of irregularly spaced data is possible with the [akima](https://cran.r-project.org/package=akima) package: `aspline()` for univariate data, `bicubic()` or `interp()` for data on a 2D rectangular domain. (This package is distributed under ACM license and not available for commercial use.)
-   Package [signal](https://cran.r-project.org/package=signal) contains several *filters* to smooth discrete data, notably `interp1()` for linear, spline, and cubic interpolation, `pchip()` for piecewise cubic Hermite interpolation, and `sgolay()` for Savitzky-Golay smoothing.
-   Package [pracma](https://cran.r-project.org/package=pracma) provides barycentric Lagrange interpolation (in 1 and 2 dimensions) in `barylag()` resp. `barylag2d()`, 1-dim. akima in `akimaInterp()`, and interpolation and approximation of data with rational functions, i.e. in the presence of singularities, in `ratinterp()` and `rationalfit()`.
-   The [interp](https://cran.r-project.org/package=interp) package provides bivariate data interpolation on regular and irregular grids, either linear or using splines. Currently the piecewise linear interpolation part is implemented. (It is intended to provide a free replacement for the ACM licensed `akima::interp` and `tripack::tri.mesh` functions.)
-   Package [chebpol](https://cran.r-project.org/package=chebpol) contains methods for creating multivariate Chebyshev and multilinear interpolation on regular grids, e.g. the Floater-Hormann barycenter method, or polyharmonic splines for scattered data.
-   [tripack](https://cran.r-project.org/package=tripack) for triangulation of irregularly spaced data is a constrained two-dimensional Delaunay triangulation package providing both triangulation and generation of Voronoi mosaics of irregular spaced data.
-   `sinterp()` in package [stinepack](https://cran.r-project.org/package=stinepack) realizes interpolation based on piecewise rational functions by applying Stineman's algorithm. The interpolating function will be monotone in regions where the specified points change monotonically.
-   `Schumaker()` in package [schumaker](https://cran.r-project.org/package=schumaker) implements shape-preserving splines, guaranteed to be monotonic resp. concave or convex if the data is monotonic, concave, or convex.
-   [ADPF](https://cran.r-project.org/package=ADPF) uses least-squares polynomial regression and statistical testing to improve Savitzky-Golay smoothing.
-   Package [conicfit](https://cran.r-project.org/package=conicfit) provides several (geometric and algebraic) algorithms for fitting circles, ellipses, and conics in general.

**Root Finding and Fixed Points**

`uniroot()`, implementing the Brent-Decker algorithm, is the basic routine in R to find roots of univariate functions. There are implementations of the bisection algorithm in several contributed packages. For root finding with higher precision there is function `unirootR()` in the multi-precision package [Rmpfr](https://cran.r-project.org/package=Rmpfr). And for finding roots of multivariate functions see the following two packages:

-   For solving nonlinear systems of equations the [BB](https://cran.r-project.org/package=BB) package provides (non-monotone) Barzilai-Borwein spectral methods in `sane()`, including a derivative-free variant in `dfsane()`, and multi-start features with sensitivity analysis.
-   Package [nleqslv](https://cran.r-project.org/package=nleqslv) solves nonlinear systems of equations using alternatively the Broyden or Newton method, supported by strategies such as line searches or trust regions.
-   [ktsolve](https://cran.r-project.org/package=ktsolve) defines a common interface for solving a set of equations with `BB` or `nleqslv`.
-   Package [FixedPoint](https://cran.r-project.org/package=FixedPoint) provides algorithms for finding fixed point vectors. These algorithms include Anderson acceleration, epsilon extrapolation methods, and minimal polynomial methods.

**Discrete Mathematics and Number Theory**

Not so many functions are available for computational number theory. Note that integers in double precision can be represented exactly up to `2^53 - 1`, above that limit a multi-precision package such as [gmp](https://cran.r-project.org/package=gmp) is needed, see below.

-   Package [numbers](https://cran.r-project.org/package=numbers) provides functions for factorization, prime numbers, twin primes, primitive roots, modular inverses, extended GCD, etc. Included are some number-theoretic functions like divisor functions or Euler's Phi function.
-   [contfrac](https://cran.r-project.org/package=contfrac) contains various utilities for evaluating continued fractions and partial convergents.
-   [magic](https://cran.r-project.org/package=magic) creates and investigates magical squares and hypercubes, including functions for the manipulation and analysis of arbitrarily dimensioned arrays.
-   Package [freegroup](https://cran.r-project.org/package=freegroup) provides functionality for manipulating elements of a free group including juxtaposition, inversion, multiplication by a scalar, power operations, and Tietze forms.
-   The [partitions](https://cran.r-project.org/package=partitions) package enumerates additive partitions of integers, including restricted and unequal partitions.
-   [permutations](https://cran.r-project.org/package=permutations) treats permutations as invertible functions of finite sets and includes several mathematical operations on them.
-   Package [combinat](https://cran.r-project.org/package=combinat) generates all permutations or all combinations of a certain length of a set of elements (i.e. a vector); it also computes binomial coefficients.
-   Package [arrangements](https://cran.r-project.org/package=arrangements) provides generators and iterators for permutations, combinations and partitions. The iterators allow users to generate arrangements in a fast and memory efficient manner. Permutations and combinations can be drawn with/without replacement and support multisets.
-   [RcppAlgos](https://cran.r-project.org/package=RcppAlgos/index.html) provides flexible functions for generating combinations or permutations of a vector with or without constraints. The extension package [bigIntegerAlgos](../packages/bigIntegerAlgos) features a quadratic sieve algorithm for completely factoring large integers.
-   Package [Zseq](https://cran.r-project.org/package=Zseq) generates well-known integer sequences; the 'gmp' package is adopted for computing with arbitrarily large numbers. Every function has on its help page a hyperlink to the corresponding entry in the On-Line Encyclopedia of Integer Sequences ( [OEIS](https://oeis.org/)).

**Multi-Precision Arithmetic and Symbolic Mathematics**

-   Multiple precision arithmetic is available in R through package [gmp](https://cran.r-project.org/package=gmp) that interfaces to the GMP C library. Examples are factorization of integers, a probabilistic prime number test, or operations on big rationals -- for which linear systems of equations can be solved.
-   Multiple precision floating point operations and functions are provided through package [Rmpfr](https://cran.r-project.org/package=Rmpfr) using the MPFR and GMP libraries. Special numbers and some special functions are included, as well as routines for root finding, integration, and optimization in arbitrary precision.
-   [Brobdingnag](https://cran.r-project.org/package=Brobdingnag) handles very large numbers by holding their logarithm plus a flag indicating their sign. (An excellent vignette explains how this is done using S4 methods.)
-   [VeryLargeIntegers](https://cran.r-project.org/package=VeryLargeIntegers) implements a multi-precision library that allows to store and manage arbitrarily big integers; it includes probabilistic primality tests and factorization algorithms.
-   Package [rSymPy](https://cran.r-project.org/package=rSymPy) accesses the symbolic algebra system 'SymPy' (written in Python) from R. It supports arbitrary precision computations, linear algebra and calculus, solving equations, discrete mathematics, and much more.
-   Package [Ryacas](https://cran.r-project.org/package=Ryacas) interfaces the computer algebra system 'Yacas'. It supports symbolic and arbitrary precision computations in calculus and linear algebra.

**Python Interfaces**

Python, through its modules 'NumPy', 'SciPy', 'Matplotlib', 'SymPy', and 'pandas', has elaborate and efficient numerical and graphical tools available.

-   [reticulate](https://cran.r-project.org/package=reticulate) is an R interface to Python modules, classes, and functions. When calling Python in R data types are automatically converted to their equivalent Python types; when values are returned from Python to R they are converted back to R types. This package from the RStudio team is a kind of standard for calling Python from R.
-   R package [rPython](https://cran.r-project.org/package=rPython/index.html) permits calls from R to Python, while [RPy](http://sourceforge.net/projects/rpy/) (with Python module 'rpy2') interfaces R from Python. [SnakeCharmR](../packages/SnakeCharmR) is a fork of 'rPython' with several fixes and improvements.
-   [PythonInR](https://cran.r-project.org/package=PythonInR) is another package to interact with Python from within R. It provides Python classes for vectors, matrices and data.frames which allow an easy conversion from R to Python and back.
-   [feather](https://cran.r-project.org/package=feather) provides bindings to read and write feather files, a lightweight binary data store designed for maximum speed. This storage format can also be accessed in Python, Julia, or Scala.
-   [findpython](https://cran.r-project.org/package=findpython) is a package designed to find an acceptable Python binary in the path, incl. minimum version or required modules.
-   'pyRserve' is a Python module for connecting Python to an R process running [Rserve](https://cran.r-project.org/package=Rserve) as an RPC gateway. This R process can run on a remote machine, variable access and function calls will be delegated through the network.
-   [XRPython](https://cran.r-project.org/package=XRPython/index.html) (and 'XRJulia') are based on John Chambers' [XR](../packages/XR) package and his "Extending R" book and allow for a very structured integration of R with Python resp. Julia.

<!-- -->

-   Note that [SageMath](http://www.sagemath.org/) is a free open source mathematics system based on Python, allowing to run R functions, but also providing access to Maxima, GAP, FLINT, and many more math programs. SageMath can be downloaded or used through a Web interface at [CoCalc](https://cocalc.com/).

**MATLAB, Octave, Julia, and other Interfaces**

Interfaces to numerical computation software such as MATLAB (commercial) or Octave (free) will be important when solving difficult numerical problems.

-   The [matlab](https://cran.r-project.org/package=matlab/index.html) emulation package contains about 30 simple functions, replicating MATLAB functions, using the respective MATLAB names and being implemented in pure R. (See also [pracma](../packages/pracma) for many more mathematical functions designed with MATLAB in mind.)
-   Package [R.matlab](https://cran.r-project.org/package=R.matlab) provides tools to read and write MAT files, which is the MATLAB data format. It also enables a one-directional interface with a MATLAB process, sending and retrieving objects through a TCP/IP connection.

Julia is "a high-level, high-performance dynamic programming language for numerical computing", which makes it interesting for optimization problems and other demanding scientific computations in R.

-   The Julia interface of the [XRJulia](https://cran.r-project.org/package=XRJulia) package by John Chambers provides direct analogues to Julia function calls. A 'juliaExamples' package is available on Github.
-   [JuliaCall](https://cran.r-project.org/package=JuliaCall) provides seamless integration between R and Julia. Using the high-level interface, the user can call any Julia function just like an R function with automatic type conversion.

The commercial programs SAS and Mathematica do have facilities to call R functions. Here is another Computer Algebra System (CAS) in Pure Mathematics that can be called from R.

-   Package [m2r](https://cran.r-project.org/package=m2r) provides a persistent interface to Macauley2, an extended software program supporting research in algebraic geometry and commutative algebra. Macauley2 has to be installed independently, otherwise a Macauley2 process in the cloud will be instantiated.

### CRAN packages:

-   [ADPF](https://cran.r-project.org/package=ADPF)
-   [akima](https://cran.r-project.org/package=akima)
-   [appell](https://cran.r-project.org/package=appell)
-   [arrangements](https://cran.r-project.org/package=arrangements)
-   [BB](https://cran.r-project.org/package=BB)
-   [Bessel](https://cran.r-project.org/package=Bessel)
-   [bigIntegerAlgos](https://cran.r-project.org/package=bigIntegerAlgos)
-   [Brobdingnag](https://cran.r-project.org/package=Brobdingnag)
-   [chebpol](https://cran.r-project.org/package=chebpol)
-   [combinat](https://cran.r-project.org/package=combinat)
-   [conicfit](https://cran.r-project.org/package=conicfit)
-   [contfrac](https://cran.r-project.org/package=contfrac)
-   [cubature](https://cran.r-project.org/package=cubature)
-   [Deriv](https://cran.r-project.org/package=Deriv) (core)
-   [eigeninv](https://cran.r-project.org/package=eigeninv)
-   [elliptic](https://cran.r-project.org/package=elliptic)
-   [expint](https://cran.r-project.org/package=expint)
-   [expm](https://cran.r-project.org/package=expm)
-   [fastGHQuad](https://cran.r-project.org/package=fastGHQuad)
-   [feather](https://cran.r-project.org/package=feather)
-   [features](https://cran.r-project.org/package=features)
-   [findpython](https://cran.r-project.org/package=findpython)
-   [FixedPoint](https://cran.r-project.org/package=FixedPoint)
-   [fourierin](https://cran.r-project.org/package=fourierin)
-   [freegroup](https://cran.r-project.org/package=freegroup)
-   [gaussquad](https://cran.r-project.org/package=gaussquad)
-   [geigen](https://cran.r-project.org/package=geigen)
-   [gmp](https://cran.r-project.org/package=gmp)
-   [gsl](https://cran.r-project.org/package=gsl)
-   [hypergeo](https://cran.r-project.org/package=hypergeo)
-   [interp](https://cran.r-project.org/package=interp)
-   [irlba](https://cran.r-project.org/package=irlba)
-   [JuliaCall](https://cran.r-project.org/package=JuliaCall)
-   [ktsolve](https://cran.r-project.org/package=ktsolve)
-   [lamW](https://cran.r-project.org/package=lamW)
-   [logOfGamma](https://cran.r-project.org/package=logOfGamma)
-   [m2r](https://cran.r-project.org/package=m2r)
-   [magic](https://cran.r-project.org/package=magic)
-   [MASS](https://cran.r-project.org/package=MASS)
-   [matlab](https://cran.r-project.org/package=matlab)
-   [Matrix](https://cran.r-project.org/package=Matrix) (core)
-   [matrixcalc](https://cran.r-project.org/package=matrixcalc)
-   [MonoPoly](https://cran.r-project.org/package=MonoPoly)
-   [mpoly](https://cran.r-project.org/package=mpoly)
-   [multipol](https://cran.r-project.org/package=multipol)
-   [mvQuad](https://cran.r-project.org/package=mvQuad)
-   [nleqslv](https://cran.r-project.org/package=nleqslv)
-   [numbers](https://cran.r-project.org/package=numbers)
-   [numDeriv](https://cran.r-project.org/package=numDeriv) (core)
-   [onion](https://cran.r-project.org/package=onion)
-   [optR](https://cran.r-project.org/package=optR)
-   [orthopolynom](https://cran.r-project.org/package=orthopolynom)
-   [Pade](https://cran.r-project.org/package=Pade)
-   [partitions](https://cran.r-project.org/package=partitions)
-   [permutations](https://cran.r-project.org/package=permutations)
-   [polyCub](https://cran.r-project.org/package=polyCub)
-   [polynom](https://cran.r-project.org/package=polynom) (core)
-   [PolynomF](https://cran.r-project.org/package=PolynomF)
-   [pracma](https://cran.r-project.org/package=pracma) (core)
-   [PRIMME](https://cran.r-project.org/package=PRIMME)
-   [PythonInR](https://cran.r-project.org/package=PythonInR)
-   [QZ](https://cran.r-project.org/package=QZ)
-   [R.matlab](https://cran.r-project.org/package=R.matlab)
-   [R2Cuba](https://cran.r-project.org/package=R2Cuba)
-   [rARPACK](https://cran.r-project.org/package=rARPACK)
-   [Rcpp](https://cran.r-project.org/package=Rcpp)
-   [RcppAlgos](https://cran.r-project.org/package=RcppAlgos)
-   [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo)
-   [RcppEigen](https://cran.r-project.org/package=RcppEigen)
-   [reticulate](https://cran.r-project.org/package=reticulate)
-   [Rlinsolve](https://cran.r-project.org/package=Rlinsolve)
-   [Rmpfr](https://cran.r-project.org/package=Rmpfr)
-   [rmumps](https://cran.r-project.org/package=rmumps)
-   [RootsExtremaInflections](https://cran.r-project.org/package=RootsExtremaInflections)
-   [rPython](https://cran.r-project.org/package=rPython)
-   [Rserve](https://cran.r-project.org/package=Rserve)
-   [RSpectra](https://cran.r-project.org/package=RSpectra)
-   [rSymPy](https://cran.r-project.org/package=rSymPy)
-   [Ryacas](https://cran.r-project.org/package=Ryacas)
-   [schumaker](https://cran.r-project.org/package=schumaker)
-   [signal](https://cran.r-project.org/package=signal)
-   [SimplicialCubature](https://cran.r-project.org/package=SimplicialCubature)
-   [SnakeCharmR](https://cran.r-project.org/package=SnakeCharmR)
-   [SolveLS](https://cran.r-project.org/package=SolveLS)
-   [SparseGrid](https://cran.r-project.org/package=SparseGrid)
-   [SparseM](https://cran.r-project.org/package=SparseM)
-   [SphericalCubature](https://cran.r-project.org/package=SphericalCubature)
-   [ssvd](https://cran.r-project.org/package=ssvd)
-   [statmod](https://cran.r-project.org/package=statmod)
-   [stinepack](https://cran.r-project.org/package=stinepack)
-   [svd](https://cran.r-project.org/package=svd)
-   [tripack](https://cran.r-project.org/package=tripack)
-   [VeryLargeIntegers](https://cran.r-project.org/package=VeryLargeIntegers)
-   [XR](https://cran.r-project.org/package=XR)
-   [XRJulia](https://cran.r-project.org/package=XRJulia)
-   [XRPython](https://cran.r-project.org/package=XRPython)
-   [Zseq](https://cran.r-project.org/package=Zseq)

### Related links:

-   CRAN Task View: [DifferentialEquations](DifferentialEquations.html)
-   CRAN Task View: [Optimization](Optimization.html)
-   CRAN Task View: [TimeSeries](TimeSeries.html)
-   CRAN Task View: [HighPerformanceComputing](HighPerformanceComputing.html)
-   <span> Textbook: [Hands-On Matrix Algebra Using R](http://www.worldscientific.com/worldscibooks/10.1142/7814) </span>
-   <span> Textbook: [Introduction to Scientific Programming and Simulation Using R](http://www.ms.unimelb.edu.au/~apro@unimelb/spuRs/index.html) </span>
-   <span> Textbook: [Numerical Methods in Science and Engineering Using R](http://www.crcpress.com/product/isbn/9781439884485) </span>
-   <span> Textbook: [Computational Methods for Numerical Analysis with R](https://www.crcpress.com/Computational-Methods-for-Numerical-Analysis-with-R/II/p/book/9781498723633) </span>
-   [R and MATLAB](http://www.math.umaine.edu/~hiebeler/comp/matlabR.html)
-   [Abramowitz and Stegun. Handbook of Mathematical Functions](http://www.nr.com/aands/)
-   [Numerical Recipes: The Art of Numerical Computing](http://www.nrbook.com/a/bookcpdf.php)
-   [E. Weisstein's Wolfram MathWorld](http://mathworld.wolfram.com/)
