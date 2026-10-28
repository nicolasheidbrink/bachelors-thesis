"""
This script was generated entirely by Claude Code. It calculates the distance 
when the intermediary densities are determined not just to be any copula densities,
but to be convex combinations of the two endpoints:
    c_t = (1 - t) * c_0 + t * c_1,     t in [0, 1].

This value is an upper bound to the copula-constrained Wasserstein because 
the linear combination is a valid interpolation satisfying the copula constraint. 


=====================================
Copula fade-path Wasserstein distance
=====================================

How the length is computed
--------------------------
Benamou-Brenier / Otto: the length of a curve c_t of probability densities is

    L = int_0^1 v(t) dt
    v(t)^2 = min_{w}{ int c_t|w|^2  :  d_t c_t + div(c_t w) = 0 }
           = int c_t |grad phi_t|^2      where  -div(c_t grad phi_t) = d_t c_t

with a no-flux (Neumann) boundary condition. The minimising w is always a
gradient field (any divergence-free remainder only adds energy), which is
why the whole problem reduces to ONE linear elliptic solve per value of t.
Discretised by finite volumes, that solve is literally a resistor network:
grid cells are nodes, cell faces are resistors whose conductance is the
local density (the "mobility"), and the equation is Kirchhoff's law.

Layout of this file
-------------------
This module is self-contained and holds BOTH layers:

  Part I  (sections 1-5) -- the generic, domain-agnostic solver. It knows
      nothing about copulas: it takes cell-mass grids on a rectangle and
      returns lengths/distances. Pinned by `test_w2_fade_path.py`.
      (This was a separate module, `w2_fade_path.py`, until 2026-08-26; it
      was folded in here because the copula adapter below was its only
      caller. Every public name is unchanged, so `W.as_grid(...)` from the
      old module is now just `as_grid(...)`.)

  Part II (sections 6-10) -- the copula-specific adapter: it turns a
      `load_target` / `MixtureCopula` object into the cell-MASS grids Part I
      requires, and `fade_distance` ties the two together.

Input contract for Part I
-------------------------
Every "grid" in Part I is an (N, N) array of CELL MASSES (i.e. already
integrated over each grid cell), NOT density values sampled at points. This
matters: distributions with singularities or sharp ridges (a Gaussian
copula's corner blow-up, for instance) are badly misrepresented by
point-sampling. Use `as_grid(...)` to go from a parametric density object
to a correct cell-mass grid -- it does this properly, with an exact path
(via a supplied CDF) when available and a quadrature fallback otherwise.
Every grid must be nonnegative and sum to 1 (up to numerical tolerance).
Part II's `copula_grid(...)` is the copula-aware wrapper around exactly this.

Two failure modes worth knowing about (both are properties of the MATH, not
bugs in this code):
  * If density0 and density1 have near-disjoint support, c_t passes through
    regions of near-zero mobility over a large fraction of the domain. The
    result is still a valid finite number (a small numerical floor keeps the
    solver well posed) but reflects a genuinely expensive path -- a warning
    fires when this fraction is large; treat a very large L as real information
    about limited overlap, not a numerical artefact. (Note that `fade_distance`
    deliberately replaces this warning with its own; see `overlap_diagnostic`.)
  * The Gauss-Legendre-in-t quadrature (`n_quad`) assumes v(t) is smooth or
    has at worst an integrable endpoint singularity, tamed by the built-in
    t = 1 - s^2 substitution. This is exactly what a copula-vs-independence
    endpoint blow-up looks like; wilder behaviour may need a larger n_quad.

Why the adapter needs analytic CDFs
-----------------------------------
Part I wants cell masses, not point-sampled densities, and most copula
densities blow up at one or both corners of [0,1]^2 (Gaussian, Clayton,
Gumbel, Joe, Student-t all do). Point-sampling or even Gauss-Legendre
quadrature misrepresents those corner cells badly. So Part II supplies an
exact joint CDF for every family in copula_families.py -- cell masses are
then EXACT finite differences of C at the cell corners, singularity and all:

  Archimedean (Clayton/Gumbel/Frank/Joe) and Independence : closed form
  Gaussian                                                : bivariate normal
      CDF by Gauss-Legendre on the conditional integral (~1e-15 accurate)
  Student-t                                               : bivariate t CDF
      by the same conditional integral under the x = sqrt(nu)*tan(phi)
      substitution, which maps R onto a finite interval and makes the
      integrand analytic (~1e-13 accurate; verified against scipy's
      multivariate_t.cdf, which is a randomised Genz QMC routine whose own
      ~1e-4 noise makes it UNUSABLE here -- differencing four noisy CDF
      values into one small cell mass amplifies that noise enormously)
  MixtureCopula                                           : weighted sum of
      the component CDFs (recursively, so nested mixtures work)
  Mirrored / Transposed / Khoudraji / Glued               : assembled from
      the wrapped copulas' CDFs by the defining formula, so nothing is
      transcribed and these nest freely too. For a GluedCopula this route is
      not merely more accurate but necessary: its density JUMPS across the
      gluing seam, so the pdf fallback is wrong in every cell the seam
      crosses (measured up to 49% off), not just near a corner.

Two spaces
----------
`space="u"` (default) works directly in copula space [0,1]^2 -- this is the
space the copula-constrained problem is posed in, so it is what you want to
compare against a trained model's copula-domain action.

`space="x"` works in the lifted R^2 density space that
copula_constrained_wasserstein_velocity_field.py actually trains in, i.e.
f = c(Phi_sigma(x), Phi_sigma(y)) * phi_sigma(x) * phi_sigma(y), on a
truncated box [-k*sigma, k*sigma]^2. The x-space CDF is exact too, since it
is just C(Phi_sigma(x), Phi_sigma(y)).

These two numbers are NOT equal and are not meant to be: u = Phi_sigma(x) is
a measure-preserving change of variables, but it is not an isometry, so the
Euclidean transport cost differs between the two spaces. Compare against
whichever space the quantity you are checking was measured in.

Usage
-----
    from distance_calculation_campaign.copula_families import load_target
    from distance_calculation_campaign.fade_distance import fade_distance

    c0 = load_target('IndependenceCopula', None, 1.0)
    c1 = load_target('GaussianCopula', 0.9, 1.0)
    res = fade_distance(c0, c1)              # -> dict, res['L'] is the metric
    res = fade_distance(c0, c1, compute_w2=True)   # + unconstrained W2 floor

Running the module directly runs the validation suite and nothing else
(from the project root, since the imports are package-qualified):

    python -m distance_calculation_campaign.fade_distance     # -> self_check()

See copula_wasserstein_bounds_example.ipynb for a worked example.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.special import roots_legendre
from scipy.stats import norm, t as _t_dist

from distance_calculation_campaign.copula_families import (load_target, MixtureCopula, MirroredCopula,
                             TransposedCopula, KhoudrajiCopula, GluedCopula,
                             bivariate_normal_cdf, bivariate_t_cdf)


Domain = tuple  # ((u0, u1), (v0, v1))
UNIT_SQUARE: Domain = ((0.0, 1.0), (0.0, 1.0))

# u values are clipped this far away from {0, 1} before being pushed through
# an inverse CDF, so Phi^-1(0) = -inf never reaches the quadrature. The exact
# boundary values of C are written back in afterwards (`_apply_boundaries`),
# so this clipping never costs accuracy at the edges -- it only keeps the
# interior evaluation finite.
_EPS = 1e-12


# ###########################################################################
# PART I -- the generic Benamou-Brenier solver
#
# Nothing below this banner and above PART II knows what a copula is: the
# whole layer speaks in cell-mass grids on a rectangle. Keep it that way --
# `test_w2_fade_path.py` exercises it with plain Gaussian bumps, with no
# copula machinery (and no torch) involved.
# ###########################################################################


# ===========================================================================
# 1. Turning a parametric density into a cell-mass grid
# ===========================================================================

def as_grid(density, N, domain: Domain = UNIT_SQUARE, quad_order=6, cdf=None):
    """Cell-mass (N,N) grid from a parametric density.

    `density` may be:
      - a vectorised callable  pdf(U, V) -> array   (U, V from np.meshgrid), or
      - an object exposing a `.pdf(u, v)` method with the same signature.

    If a joint CDF is available -- pass it explicitly via `cdf=`, or it will
    be auto-detected as `density.cdf` -- cell masses are computed EXACTLY as
    finite differences of the CDF at cell edges. This is the preferred path
    whenever available: it costs one extra method and sidesteps every
    quadrature-accuracy question, including singularities. The CDF must
    satisfy the usual boundary conditions on `domain`: C(u0, v) = C(u, v0) = 0
    and C(u1, v1) = 1 (i.e. it is a genuine joint CDF restricted to `domain`,
    not necessarily on all of R^2).

    Otherwise, cell masses are estimated by composite Gauss-Legendre
    quadrature (`quad_order` nodes per cell per axis) -- fully vectorised,
    accurate for smooth densities. If your density has a sharp ridge or
    singularity and no CDF is available, increase `quad_order` and/or `N`
    and check convergence (see test_w2_fade_path.py for the pattern).
    """
    pdf = getattr(density, "pdf", density)
    cdf = cdf if cdf is not None else getattr(density, "cdf", None)
    (u0, u1), (v0, v1) = domain

    if cdf is not None:
        return _grid_from_cdf(cdf, N, domain)
    return _grid_from_pdf_quadrature(pdf, N, domain, quad_order)


def _grid_from_cdf(cdf, N, domain: Domain):
    (u0, u1), (v0, v1) = domain
    us = np.linspace(u0, u1, N + 1)
    vs = np.linspace(v0, v1, N + 1)
    U, V = np.meshgrid(us, vs, indexing="ij")
    C = np.asarray(cdf(U, V), dtype=float)
    if C.shape != (N + 1, N + 1):
        raise ValueError(f"cdf(U,V) must return shape {(N+1, N+1)}, got {C.shape}")
    mass = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    mass = np.clip(mass, 0.0, None)
    _check_mass(mass, "as_grid (via cdf)")
    return mass


def _grid_from_pdf_quadrature(pdf, N, domain: Domain, quad_order):
    (u0, u1), (v0, v1) = domain
    hu, hv = (u1 - u0) / N, (v1 - v0) / N

    gx, gw = roots_legendre(quad_order)          # nodes/weights on [-1,1]
    local_u = 0.5 * hu * gx                       # offsets within a cell
    local_v = 0.5 * hv * gx
    wu = 0.5 * hu * gw
    wv = 0.5 * hv * gw

    cell_u0 = u0 + hu * np.arange(N)              # left edge of each cell
    cell_v0 = v0 + hv * np.arange(N)
    u_nodes = (cell_u0[:, None] + hu / 2 + local_u[None, :]).ravel()  # (N*q,)
    v_nodes = (cell_v0[:, None] + hv / 2 + local_v[None, :]).ravel()

    U, V = np.meshgrid(u_nodes, v_nodes, indexing="ij")
    P = np.asarray(pdf(U, V), dtype=float)
    if P.shape != U.shape:
        raise ValueError(f"pdf(U,V) must return shape {U.shape}, got {P.shape}")

    q = quad_order
    P = P.reshape(N, q, N, q)
    mass = np.einsum("ikjl,k,l->ij", P, wu, wv)
    mass = np.clip(mass, 0.0, None)
    _check_mass(mass, "as_grid (via pdf quadrature)")
    return mass


def grid_from_samples(samples_u, samples_v, N, domain: Domain = UNIT_SQUARE, weights=None):
    """Fallback: cell-mass grid from raw samples (histogram estimator).

    Coarser than as_grid() -- only use this when no pdf/cdf is available,
    e.g. the parametric object only supports .sample(). Prefer a larger
    number of samples relative to N**2 (rule of thumb: >= 50 samples per
    cell on average) or the grid will be noisy.
    """
    (u0, u1), (v0, v1) = domain
    H, _, _ = np.histogram2d(
        samples_u, samples_v, bins=N, range=[[u0, u1], [v0, v1]], weights=weights,
    )
    mass = H / H.sum()
    _check_mass(mass, "grid_from_samples")
    return mass


def _check_mass(mass, where):
    total = mass.sum()
    if not np.isfinite(total) or abs(total - 1.0) > 1e-3:
        warnings.warn(f"{where}: cell masses sum to {total:.6f}, expected ~1. "
                       f"Check that the density is normalised on `domain`.")
    if np.any(mass < -1e-9):
        warnings.warn(f"{where}: negative cell mass encountered "
                       f"(min {mass.min():.3e}); clipped to 0.")


# ===========================================================================
# 2. The core solve: one linear elliptic PDE, as a resistor network
# ===========================================================================

def _faces_and_scale(N, hu, hv):
    """Face index pairs and their finite-volume conductance scale factor.

    u-faces (normal in the u direction): scale = hv/hu  (face length / spacing)
    v-faces (normal in the v direction): scale = hu/hv
    """
    idx = np.arange(N * N).reshape(N, N)
    u_faces = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])
    v_faces = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
    faces = np.concatenate([u_faces, v_faces], axis=1)
    scale = np.concatenate([
        np.full(u_faces.shape[1], hv / hu),
        np.full(v_faces.shape[1], hu / hv),
    ])
    return faces, scale


def metric_speed(c_density, f_mass, faces, scale, mobility_floor_rel=1e-8,
                  low_mobility_thresh=1e-3):
    """Solve  -div(c grad phi) = f  (Neumann BC) and return
    (speed, phi, floored_frac, low_mobility_frac).

    c_density : (N,N) array -- the MOBILITY, in DENSITY units (mass / cell_area).
                Do not pass raw cell masses here -- convert first
                (c_density = mass_grid / cell_area). Getting this wrong silently
                rescales the whole result; there is no way to detect it from
                shape alone, which is exactly why every caller in this module
                does the conversion in one obvious place -- keep that pattern
                if you call this function directly.
    f_mass    : (N,N) array -- the source, in MASS units already integrated
                over each cell (e.g. a cell-mass difference), NOT multiplied
                by cell_area again. Must have ~zero total sum.
    faces     : from _faces_and_scale
    scale     : from _faces_and_scale
    mobility_floor_rel : numerical floor applied to c_density as
        `max(c_density, floor_rel * c_density.max())`, purely to keep the
        linear system well posed. `floored_frac` reports what fraction of
        total conductance the floor altered -- this is a numerical-safety
        diagnostic, NOT a proxy for "is this a sensible problem": a solve can
        be numerically fine (floored_frac ~ 0) while still being physically
        expensive, because most of the domain has genuinely tiny mobility.
    low_mobility_thresh : separate, physical diagnostic. `low_mobility_frac`
        reports what fraction of CELLS (unweighted) have density below
        `low_mobility_thresh * c_density.max()`, BEFORE flooring. A large
        value (e.g. > 0.3) means density0/density1 have limited overlap along
        much of the domain -- the fade path is legitimately expensive there,
        which is real information, not a bug (see module docstring). For
        copula inputs specifically this measure is the wrong one; see
        `overlap_diagnostic` in Part II.
    """
    N = c_density.shape[0]
    cf = c_density.ravel()
    cmax = max(cf.max(), 1e-300)
    low_mobility_frac = float(np.mean(cf < low_mobility_thresh * cmax))

    floor = mobility_floor_rel * cmax
    cf_reg = np.maximum(cf, floor)
    floored_frac = float(np.sum(cf_reg - cf)) / max(float(cf_reg.sum()), 1e-300)

    a, b = faces
    w = 0.5 * (cf_reg[a] + cf_reg[b]) * scale

    n = N * N
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([a, b, b, a])
    vals = np.concatenate([w, w, -w, -w])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    rhs = f_mass.ravel()          # already mass units -- no cell_area factor
    phi = np.zeros(n)
    phi[1:] = spla.spsolve(A[1:, 1:].tocsc(), rhs[1:])   # pin phi[0] = 0

    energy = float(phi @ rhs)
    speed = np.sqrt(max(energy, 0.0))
    return speed, phi.reshape(N, N), floored_frac, low_mobility_frac


# ===========================================================================
# 3. Path length: the linear-mixture fade path, and the general case
# ===========================================================================

def fade_path_length(grid0, grid1, domain: Domain = UNIT_SQUARE, n_quad=48,
                      mobility_floor_rel=1e-8, return_profile=False):
    """W2 length of  c_t = (1-t)*grid0 + t*grid1,  t in [0,1].

    This is NOT the W2 geodesic distance; it is how far you travel if you
    insist on that specific (simple, family-preserving) path. See
    `path_length` for arbitrary paths, and `w2_exact` for the geodesic.

    grid0, grid1 : (N,N) cell-mass arrays (same N, same domain, each summing to 1).
    n_quad       : Gauss-Legendre nodes for the t-integral, via t = 1 - s^2
                   (tames an endpoint speed blow-up if density support shrinks
                   to zero at either endpoint, e.g. corner singularities).
    """
    grid0 = np.asarray(grid0, float)
    grid1 = np.asarray(grid1, float)
    if grid0.shape != grid1.shape:
        raise ValueError(f"grid0 {grid0.shape} and grid1 {grid1.shape} must match")
    N = grid0.shape[0]
    (u0, u1), (v0, v1) = domain
    hu, hv = (u1 - u0) / N, (v1 - v0) / N
    cell_area = hu * hv

    f = grid1 - grid0
    f -= f.mean()                                  # enforce exact zero mean
    faces, scale = _faces_and_scale(N, hu, hv)

    gx, gw = roots_legendre(n_quad)
    s = 0.5 * (gx + 1.0)
    w = 0.5 * gw
    ts = 1.0 - s ** 2                               # tame endpoint blow-up

    speeds = np.empty(n_quad)
    floors = np.empty(n_quad)
    low_mob = np.empty(n_quad)
    for i, t in enumerate(ts):
        c_density = ((1 - t) * grid0 + t * grid1) / cell_area
        speeds[i], _, floors[i], low_mob[i] = metric_speed(
            c_density, f, faces, scale, mobility_floor_rel)

    L = float(np.sum(w * speeds * 2.0 * s))
    max_floor = float(floors.max())
    max_low_mob = float(low_mob.max())
    if max_low_mob > 0.3:
        warnings.warn(
            f"fade_path_length: up to {max_low_mob:.0%} of the domain has "
            f"near-zero density at some point along the path -- density0/"
            f"density1 likely have limited overlap (near-disjoint support). "
            f"The result is still a valid finite length, but reflects a "
            f"genuinely expensive path -- treat a very large L as real, not "
            f"a numerical artefact (see module docstring)."
        )
    if return_profile:
        return L, ts, speeds, max_floor
    return L


def path_length(path_fn, N, domain: Domain = UNIT_SQUARE, n_quad=48,
                 dpath_fn=None, dt_fd=1e-4, mobility_floor_rel=1e-8):
    """W2 length of an ARBITRARY path of cell-mass grids, c_t = path_fn(t).

    path_fn(t)  -> (N,N) cell-mass grid at time t in [0,1].
    dpath_fn(t) -> (N,N) array, d/dt of path_fn at t. If omitted, estimated
                   by central finite differences (dt_fd), which costs two
                   extra evaluations of path_fn per quadrature node -- supply
                   an analytic derivative when you have one.

    Use this instead of fade_path_length() for any path other than the plain
    linear mixture (e.g. geodesics under some other reference metric, or a
    displacement-interpolation-like schedule you construct by hand).
    """
    (u0, u1), (v0, v1) = domain
    hu, hv = (u1 - u0) / N, (v1 - v0) / N
    cell_area = hu * hv
    faces, scale = _faces_and_scale(N, hu, hv)

    def deriv(t):
        if dpath_fn is not None:
            return np.asarray(dpath_fn(t), float)
        tp, tm = min(t + dt_fd, 1.0), max(t - dt_fd, 0.0)
        return (path_fn(tp) - path_fn(tm)) / (tp - tm)

    gx, gw = roots_legendre(n_quad)
    s = 0.5 * (gx + 1.0)
    w = 0.5 * gw
    ts = 1.0 - s ** 2

    speeds = np.empty(n_quad)
    max_low_mob = 0.0
    for i, t in enumerate(ts):
        c_density = np.asarray(path_fn(t), float) / cell_area
        f = deriv(t)                    # d(mass)/dt -- already correct rhs units
        f = f - f.mean()
        sp_i, _, _, lm_i = metric_speed(c_density, f, faces, scale, mobility_floor_rel)
        speeds[i] = sp_i
        max_low_mob = max(max_low_mob, lm_i)

    if max_low_mob > 0.3:
        warnings.warn(
            f"path_length: up to {max_low_mob:.0%} of the domain has "
            f"near-zero density at some point along the path; a large result "
            f"may be real, not a numerical artefact (see module docstring)."
        )
    return float(np.sum(w * speeds * 2.0 * s))


# ===========================================================================
# 4. True W2 (geodesic) distance -- exact transportation LP, for reference
# ===========================================================================

def w2_exact(grid0, grid1, domain: Domain = UNIT_SQUARE, mass_floor=1e-13):
    """Exact discrete W2 distance (squared-Euclidean cost) via linprog/HiGHS.

    No entropic regularisation, no epsilon to tune -- this is ground truth
    for the discretised problem, and fade_path_length(...) >= w2_exact(...)
    always (equality only if the linear mixture happens to already be the
    geodesic, which is generically false).

    Cost grows fast with grid resolution (dense n x m LP): keep grid0/grid1
    modest, e.g. N <= 32, or subsample/coarsen before calling this. Use a
    much finer grid for fade_path_length (the PDE solve is cheap) and only
    a coarse one here for the reference distance.
    """
    from scipy.optimize import linprog

    grid0 = np.asarray(grid0, float)
    grid1 = np.asarray(grid1, float)
    N = grid0.shape[0]
    (u0, u1), (v0, v1) = domain
    z_u = u0 + (np.arange(N) + 0.5) * (u1 - u0) / N
    z_v = v0 + (np.arange(N) + 0.5) * (v1 - v0) / N
    P = np.stack(np.meshgrid(z_u, z_v, indexing="ij"), axis=-1).reshape(N * N, 2)

    a = grid0.ravel()
    a = a / a.sum()
    b_full = grid1.ravel()
    b_full = b_full / b_full.sum()

    keep = b_full > mass_floor
    Q = P[keep]
    b = b_full[keep]
    b = b / b.sum()
    b[int(np.argmax(b))] += a.sum() - b.sum()      # exact float consistency
    m = len(b)
    n = len(a)

    C = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1).ravel()
    cols = np.arange(n * m)
    A1 = sp.coo_matrix((np.ones(n * m), (np.repeat(np.arange(n), m), cols)), shape=(n, n * m))
    A2 = sp.coo_matrix((np.ones(n * m), (np.tile(np.arange(m), n), cols)), shape=(m, n * m))
    A = sp.vstack([A1, A2.tocsr()[:-1]]).tocsr()      # last row redundant
    rhs = np.concatenate([a, b[:-1]])

    res = linprog(C, A_eq=A, b_eq=rhs, bounds=(0, None), method="highs")
    if not res.success and "infeasible" in str(res.message).lower():
        # This transport LP is ALWAYS feasible by construction: a and b are
        # nonnegative and sum(a) == sum(b) is forced exactly above, so the
        # product coupling a_i b_j is a feasible witness. An "infeasible"
        # verdict is therefore a solver artefact, not a property of the
        # problem -- HiGHS's presolve mis-certifies infeasibility when the
        # masses span many orders of magnitude, which copula corner
        # singularities routinely produce (rhs entries from ~1e-13 to ~1e-2
        # for, e.g., Frank(25) vs Clayton(15) mirrored along u0). Retry once with
        # presolve off before believing it.
        res = linprog(C, A_eq=A, b_eq=rhs, bounds=(0, None), method="highs",
                      options={"presolve": False})
    if not res.success:
        raise RuntimeError(f"w2_exact LP failed: {res.message}")
    return float(np.sqrt(res.fun))


# ===========================================================================
# 5. One-call convenience wrapper (generic densities, not copulas)
# ===========================================================================

def compare(density0, density1, domain: Domain = UNIT_SQUARE,
            N_pde=128, N_lp=24, n_quad=48, quad_order=6):
    """Convenience entry point: build grids from two parametric densities and
    report both the fade-path length and the true W2 distance.

    For copulas use `fade_distance` (section 8) instead -- it takes copula
    objects, routes them through the exact-CDF grid builder, and knows about
    the u/x spaces. This one is the generic-layer equivalent.

    Returns a dict: {L, W2, ratio, v0, floored_frac, N_pde, N_lp}
    """
    grid0_pde = as_grid(density0, N_pde, domain, quad_order)
    grid1_pde = as_grid(density1, N_pde, domain, quad_order)
    L, ts, speeds, floored = fade_path_length(
        grid0_pde, grid1_pde, domain, n_quad, return_profile=True)

    # speed exactly at t=0 (quadrature nodes never land exactly on an endpoint)
    (u0, u1), (dv0, v1) = domain
    hu, hv = (u1 - u0) / N_pde, (v1 - dv0) / N_pde
    faces, scale = _faces_and_scale(N_pde, hu, hv)
    f0 = grid1_pde - grid0_pde
    f0 = f0 - f0.mean()
    v_at_0, _, _, _ = metric_speed(grid0_pde / (hu * hv), f0, faces, scale)

    grid0_lp = as_grid(density0, N_lp, domain, quad_order)
    grid1_lp = as_grid(density1, N_lp, domain, quad_order)
    W2 = w2_exact(grid0_lp, grid1_lp, domain)

    return {
        "L": L, "W2": W2, "ratio": L / W2 if W2 > 0 else float("nan"),
        "v0": v_at_0, "floored_frac": floored, "N_pde": N_pde, "N_lp": N_lp,
    }


# ###########################################################################
# PART II -- the copula adapter
#
# Everything below turns copula objects from copula_families.py into the
# cell-mass grids Part I consumes, and wraps the result in the diagnostics
# that only make sense for copulas.
# ###########################################################################


# ===========================================================================
# 6. Copula object  ->  pdf / cdf callables on [0,1]^2
# ===========================================================================

# `bivariate_normal_cdf` and `bivariate_t_cdf` -- the two non-elementary CDFs
# -- live in copula_families.py, which is the lower-level module and needed
# them on the torch side too once the Khoudraji construction arrived. They are
# imported above and re-exported here, so `CF.bivariate_normal_cdf` still
# resolves for anything that used them from this module.


def _apply_boundaries(C, U, V):
    """Write the exact copula boundary values into a numerically computed C.

    Every copula satisfies C(u,0) = C(0,v) = 0, C(u,1) = u, C(1,v) = v. The
    quadrature above is accurate but not exact there, and the whole point of
    the CDF route is that the *edge* cells (where the density is singular)
    come out right, so pin them.
    """
    C = np.where((U <= _EPS) | (V <= _EPS), 0.0, C)
    C = np.where(U >= 1.0 - _EPS, np.broadcast_to(V, C.shape), C)
    C = np.where(V >= 1.0 - _EPS, np.broadcast_to(U, C.shape), C)
    C = np.where((U >= 1.0 - _EPS) & (V >= 1.0 - _EPS), 1.0, C)
    return C


class NoAnalyticCDF(Exception):
    """Raised when a copula has no CDF implemented here (falls back to pdf)."""


def copula_cdf_u(copula):
    """Return a vectorised `cdf(U, V) -> array` for a copula object.

    U, V are arrays of the same shape (from np.meshgrid(..., indexing='ij')),
    with entries in [0,1]; the return value is C(u,v) elementwise.
    Raises NoAnalyticCDF for a family not handled here.
    """
    target = copula.target

    if target == 'MixtureCopula':
        sub = [copula_cdf_u(c) for c in copula.components]
        wts = copula.weights.detach().cpu().numpy().astype(float)

        def cdf(U, V):
            return sum(w * f(U, V) for w, f in zip(wts, sub))
        return cdf

    if target == 'KhoudrajiCopula':
        # K(u,v) = C1(u^(1-a1), v^(1-a2)) * C2(u^a1, v^a2) -- the defining
        # formula, so nothing to transcribe and no new quadrature
        a1, a2 = copula.shape
        f1, f2 = copula_cdf_u(copula.base1), copula_cdf_u(copula.base2)

        def cdf(U, V):
            return f1(U ** (1 - a1), V ** (1 - a2)) * f2(U ** a1, V ** a2)
        return cdf

    if target == 'GluedCopula':
        # eqs. (3)-(4) of Siburg-Stoimenov, assembled from the two bases'
        # CDFs. This route matters more here than for the other composites:
        # a glued density is DISCONTINUOUS across the seam and inherits the
        # bases' corner singularities all along it, so the pdf-quadrature
        # fallback would be wrong in every cell the seam crosses, not just
        # near the four corners of the square.
        i, th = copula.index, float(copula.theta)
        f1, f2 = copula_cdf_u(copula.base1), copula_cdf_u(copula.base2)

        def cdf(U, V):
            W, Z = (U, V) if i == 0 else (V, U)
            W1 = np.clip(W / th, 0.0, 1.0)
            W2 = np.clip((W - th) / (1.0 - th), 0.0, 1.0)
            pair = lambda w: ((w, Z) if i == 0 else (Z, w))
            out = np.where(W <= th,
                           th * f1(*pair(W1)),
                           (1.0 - th) * f2(*pair(W2)) + th * Z)
            return np.clip(out, 0.0, 1.0)
        return cdf

    if target == 'TransposedCopula':
        # C_T(u,v) = C(v,u): the swap is measure-preserving and leaves the
        # rectangle [0,u] x [0,v] a rectangle, so unlike the mirrored case
        # there are no boundary terms to add back
        base = copula_cdf_u(copula.base)
        if not copula.swapped:
            return base

        def cdf(U, V):
            return base(V, U)
        return cdf

    if target == 'MirroredCopula':
        # C_mir from the base C, by integrating c(u',v') over the reflected
        # rectangle: mirroring u contributes the C(1,v)=v term, mirroring v the
        # C(u,1)=u term, and mirroring both subtracts the doubly counted 1.
        base = copula_cdf_u(copula.base)
        fu, fv = copula.flip

        def cdf(U, V):
            Ur, Vr = (1.0 - U if fu else U), (1.0 - V if fv else V)
            sign = (-1.0) ** (fu + fv)
            out = sign * base(Ur, Vr)
            if fu:
                out = out + V
            if fv:
                out = out + U
            if fu and fv:
                out = out - 1.0
            return np.clip(out, 0.0, 1.0)
        return cdf

    theta = copula.theta.detach().cpu().numpy() if copula.theta is not None else None

    if target == 'IndependenceCopula':
        def cdf(U, V):
            return U * V

    elif target == 'GaussianCopula':
        rho = float(theta)

        def cdf(U, V):
            a = norm.ppf(np.clip(U, _EPS, 1 - _EPS))
            b = norm.ppf(np.clip(V, _EPS, 1 - _EPS))
            return _apply_boundaries(bivariate_normal_cdf(a, b, rho), U, V)

    elif target == 'StudentTCopula':
        rho, nu = float(theta[0]), float(theta[1])

        def cdf(U, V):
            a = _t_dist.ppf(np.clip(U, _EPS, 1 - _EPS), df=nu)
            b = _t_dist.ppf(np.clip(V, _EPS, 1 - _EPS), df=nu)
            return _apply_boundaries(bivariate_t_cdf(a, b, rho, nu), U, V)

    elif target == 'ClaytonCopula':
        th = float(theta)

        def cdf(U, V):
            Uc, Vc = np.clip(U, _EPS, 1.0), np.clip(V, _EPS, 1.0)
            inner = np.clip(Uc ** (-th) + Vc ** (-th) - 1.0, 1e-300, None)
            return _apply_boundaries(inner ** (-1.0 / th), U, V)

    elif target == 'GumbelCopula':
        th = float(theta)

        def cdf(U, V):
            Uc, Vc = np.clip(U, _EPS, 1.0), np.clip(V, _EPS, 1.0)
            tmp = (-np.log(Uc)) ** th + (-np.log(Vc)) ** th
            return _apply_boundaries(np.exp(-tmp ** (1.0 / th)), U, V)

    elif target == 'FrankCopula':
        th = float(theta)

        def cdf(U, V):
            num = np.expm1(-th * U) * np.expm1(-th * V)
            return _apply_boundaries(
                -1.0 / th * np.log1p(num / np.expm1(-th)), U, V)

    elif target == 'JoeCopula':
        th = float(theta)

        def cdf(U, V):
            Ub, Vb = np.clip(1.0 - U, 0.0, 1.0), np.clip(1.0 - V, 0.0, 1.0)
            tmp = Ub ** th + Vb ** th - (Ub * Vb) ** th
            return _apply_boundaries(1.0 - tmp ** (1.0 / th), U, V)

    elif target == 'PlackettCopula':
        th = float(theta)
        eta = th - 1.0

        def cdf(U, V):
            if abs(eta) < 1e-12:            # independence limit: the only 0/0
                return _apply_boundaries(U * V, U, V)
            S = 1.0 + eta * (U + V)
            D = np.maximum(S ** 2 - 4.0 * th * eta * U * V, 0.0)
            return _apply_boundaries((S - np.sqrt(D)) / (2.0 * eta), U, V)

    else:
        raise NoAnalyticCDF(f"no analytic CDF implemented for {target!r}")

    return cdf


def copula_pdf_u(copula):
    """Vectorised `pdf(U, V) -> array`: the copula density c(u,v) on [0,1]^2,
    evaluated through the torch implementation in copula_families (float64,
    on CPU, gradients off -- this is pure numerics, nothing is trained here).
    """
    def pdf(U, V):
        u = torch.from_numpy(np.stack([np.asarray(U, float),
                                       np.asarray(V, float)], axis=-1))
        with torch.no_grad():
            return copula.density_u(u).detach().cpu().numpy().astype(float)
    return pdf


def copula_pdf_x(copula):
    """Vectorised `pdf(X, Y) -> array`: the lifted R^2 joint density
    f(x,y) = c(Phi_sigma(x), Phi_sigma(y)) * phi_sigma(x) * phi_sigma(y)."""
    def pdf(X, Y):
        x = torch.from_numpy(np.stack([np.asarray(X, float),
                                       np.asarray(Y, float)], axis=-1))
        with torch.no_grad():
            return copula.density_x(x).detach().cpu().numpy().astype(float)
    return pdf


def copula_cdf_x(copula):
    """Vectorised `cdf(X, Y)` in the lifted R^2 space. Exact given the
    u-space CDF, by Sklar: F(x,y) = C(Phi_sigma(x), Phi_sigma(y))."""
    cdf_u = copula_cdf_u(copula)
    sigma = float(copula.sigma)

    def cdf(X, Y):
        return cdf_u(norm.cdf(np.asarray(X, float) / sigma),
                     norm.cdf(np.asarray(Y, float) / sigma))
    return cdf


# ===========================================================================
# 7. Copula object  ->  cell-mass grid
# ===========================================================================

def x_domain(copula, n_sigma=6.0):
    """Truncated box for the lifted R^2 space: [-n_sigma*s, n_sigma*s]^2.
    At the default n_sigma=6 the discarded tail mass is ~1e-9, far below
    every other error in the pipeline."""
    s = float(copula.sigma) * float(n_sigma)
    return ((-s, s), (-s, s))


def copula_grid(copula, N=128, space="u", domain=None, quad_order=8,
                use_cdf=True):
    """Cell-mass (N,N) grid for a copula, ready for the Part I solver.

    space : "u" for copula space [0,1]^2, "x" for the lifted R^2 density
            space (see module docstring).
    use_cdf : take the exact CDF-differencing route when a CDF is available.
            Set False only to exercise the quadrature fallback (the two agree
            away from singular corners -- `check_cdf_consistency` checks it).
    """
    if space == "u":
        domain = UNIT_SQUARE if domain is None else domain
        pdf, cdf_fn = copula_pdf_u(copula), copula_cdf_u
    elif space == "x":
        domain = x_domain(copula) if domain is None else domain
        pdf, cdf_fn = copula_pdf_x(copula), copula_cdf_x
    else:
        raise ValueError(f"space must be 'u' or 'x', got {space!r}")

    cdf = None
    if use_cdf:
        try:
            cdf = cdf_fn(copula)
        except NoAnalyticCDF as exc:
            warnings.warn(f"{exc}; falling back to pdf quadrature, which is "
                          f"inaccurate near a density singularity.")
    return as_grid(pdf, N, domain=domain, quad_order=quad_order, cdf=cdf)


def overlap_diagnostic(grid0, grid1, rel=1e-3, quantile=0.99):
    """Fraction of cells in which BOTH endpoints are negligible.

    Since c_t = (1-t)*c_0 + t*c_1 >= min(c_0, c_1) cellwise, the cells that
    make the fade path expensive at ANY t are exactly those where that
    cellwise minimum is tiny -- so that is what this measures.

    The density scale is the `quantile`-th percentile, NOT the maximum,
    because almost every copula density is singular at a corner: a single
    cell then sets the max an order of magnitude above the bulk of the
    distribution (measured here: p99/max ~ 0.04 for a Gaussian copula at
    N=128). Part I's own low-mobility warning is max-relative, so it reports
    "36% near-zero density, likely near-disjoint support" for something as
    ordinary as independence vs. Gaussian(0.9) -- which is wrong, since
    copula densities are strictly positive on the open square and can never
    actually have disjoint support. `fade_distance` therefore suppresses that
    warning and re-raises it off this diagnostic instead.
    """
    scale = np.quantile(np.concatenate([np.asarray(grid0).ravel(),
                                        np.asarray(grid1).ravel()]), quantile)
    return float(np.mean(np.minimum(grid0, grid1) < rel * scale))


# ===========================================================================
# 8. The metric itself
# ===========================================================================

def fade_distance(copula0, copula1, N=128, n_quad=48, space="u",
                  compute_w2=False, N_w2=24, quad_order=8, use_cdf=True,
                  domain=None, return_profile=False, mobility_floor_rel=1e-8):
    """Fade-path W2 length between two copulas -- the metric this script exists for.

    copula0, copula1 : `load_target` or `MixtureCopula` instances. For
        space="x" they must share the same marginal `sigma` (they always do
        in this project -- the lift is defined against one fixed marginal).
    N        : PDE grid resolution. The elliptic solve is cheap; 128 is a
               good default, 256 is affordable if you want a convergence check.
    n_quad   : Gauss-Legendre nodes for the t-integral.
    compute_w2 : also run the exact unconstrained-W2 transportation LP on a
               coarse N_w2 grid, giving the lower end of the sandwich
               W2 <= d_constrained <= L. Costs seconds-to-minutes; keep
               N_w2 <= ~32.
    return_profile : include the speed profile v(t) in the result, which is
               what to look at when a number seems surprising -- a spike near
               an endpoint means a corner singularity is dominating the length.

    Returns a dict:
        L            the fade-path length (THE metric)
        W2           unconstrained W2 (None unless compute_w2)
        ratio        L / W2 (None unless compute_w2) -- how much the
                     copula-preserving detour costs relative to free transport
        floored_frac numerical-safety diagnostic from the solver (see
                     `metric_speed`); should be ~0
        low_mobility_frac  fraction of the domain where both copulas are
                     negligible (see `overlap_diagnostic`); large means the
                     path is genuinely expensive, not that anything is wrong
        ts, speeds   the v(t) profile (only if return_profile)
        notes        any diagnostics worth reading
    """
    if space == "x" and float(copula0.sigma) != float(copula1.sigma):
        raise ValueError("space='x' requires both copulas to share `sigma` "
                         f"(got {copula0.sigma} and {copula1.sigma})")
    if domain is None:
        domain = UNIT_SQUARE if space == "u" else x_domain(copula0)

    grid0 = copula_grid(copula0, N, space, domain, quad_order, use_cdf)
    grid1 = copula_grid(copula1, N, space, domain, quad_order, use_cdf)

    notes = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        L, ts, speeds, floored = fade_path_length(
            grid0, grid1, domain=domain, n_quad=n_quad,
            mobility_floor_rel=mobility_floor_rel, return_profile=True)
    low_frac = overlap_diagnostic(grid0, grid1)
    for wrn in caught:
        msg = str(wrn.message)
        if "near-zero density" in msg:
            # Suppressed on purpose and replaced below: Part I's version of
            # this warning is relative to the max density, which a copula corner
            # singularity (space="u") or a Gaussian tail inside the truncated box
            # (space="x") inflates for every input, making it fire almost always.
            # See `overlap_diagnostic`.
            continue
        warnings.warn(msg)
        notes.append(msg)
    if low_frac > 0.3:
        msg = (f"both copulas have negligible density over {low_frac:.0%} of "
               f"the domain, so the fade path spends much of its time moving "
               f"mass through near-empty regions")
        if space == "x":
            # In the lifted space this says nothing about the two copulas: the
            # box is [-6 sigma, 6 sigma]^2 and the SHARED Gaussian marginal
            # makes ~70% of it near-empty for any pair whatsoever. Note it,
            # don't warn about it.
            notes.append(f"{msg} -- expected for every pair in space='x', "
                         f"where the shared Gaussian marginal, not the "
                         f"copulas, empties the corners of the truncated box")
        else:
            warnings.warn(f"fade_distance: {msg}. L is still a valid finite "
                          f"length, but a large value here is real -- it is "
                          f"what the linear-mixture path genuinely costs, not "
                          f"a numerical artefact.")
            notes.append(msg)

    result = {"L": L, "W2": None, "ratio": None, "floored_frac": floored,
              "low_mobility_frac": low_frac, "space": space, "N": N,
              "n_quad": n_quad, "notes": notes}

    if compute_w2:
        g0 = copula_grid(copula0, N_w2, space, domain, quad_order, use_cdf)
        g1 = copula_grid(copula1, N_w2, space, domain, quad_order, use_cdf)
        w2 = w2_exact(g0, g1, domain=domain)
        result["W2"] = w2
        result["N_w2"] = N_w2
        result["ratio"] = L / w2 if w2 > 0 else float("nan")
        if w2 > L * (1 + 1e-6):
            # W2 <= L is a theorem; only discretisation (two different grids)
            # can break it, and only when both are nearly equal anyway.
            notes.append(f"W2 ({w2:.6f}) exceeds L ({L:.6f}) -- the two use "
                         f"different grids (N={N} vs N_w2={N_w2}); expect this "
                         f"only when the two copulas are nearly identical.")
    if return_profile:
        result["ts"], result["speeds"] = ts, speeds
    return result


def fade_path_copula(copula0, copula1, t):
    """The copula sitting at parameter t on the fade path, as a real
    `MixtureCopula` object -- so an interpolated c_t can be sampled,
    plotted, or fed to anything else in this project exactly like an
    endpoint copula. This is the object whose W2 length `fade_distance`
    measures, and its existence IS the constraint argument: a convex
    combination of copulas is a copula."""
    t = float(t)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must lie in [0,1], got {t}")
    return MixtureCopula([copula0, copula1], [1.0 - t, t])


# ===========================================================================
# 9. Consistency checks
# ===========================================================================

def _copula_label(copula):
    """Short human-readable label for a copula object, for check output.

    Only the transpose recurses (it has nothing of its own to print, so a
    bare "KhoudrajiCopula^T" would say less than the wrapper it sits on);
    the other composites name their components by family alone, which keeps
    the line short enough to read.
    """
    name = copula.target
    theta = getattr(copula, "theta", None)
    if name == 'TransposedCopula':
        return _copula_label(copula.base) + ("^T" if copula.swapped else "")
    if name == 'MirroredCopula':
        name = f"{copula.base.target}@{copula.axes_tag}"
    if name == 'KhoudrajiCopula':
        name = (f"Khoudraji({copula.base1.target[:-6]},{copula.base2.target[:-6]})"
                f"[{copula.shape[0]:g},{copula.shape[1]:g}]")
    if name == 'GluedCopula':
        # a GluedCopula's `theta` is where the cut goes, not a dependence
        # parameter, so it belongs in the name rather than after it
        name = (f"Glued({copula.base1.target[:-6]},{copula.base2.target[:-6]})"
                f"@u{copula.index}={copula.cut:g}")
        theta = None
    pretty = (np.array2string(np.asarray(theta.detach(), dtype=float),
                              precision=3, floatmode='maxprec')
              if theta is not None else None)
    return f"{name}" + (f"(theta={pretty})" if theta is not None else "")


def check_cdf_consistency(copula, N=32, rtol=2e-3, verbose=True):
    """Cross-check the analytic CDF against pdf quadrature on the same grid.

    The two routes must agree except in cells containing a density
    singularity (a copula corner), where the quadrature -- not the CDF -- is
    the wrong one. So this compares interior cells only, and is the check
    that catches a mis-transcribed CDF formula.
    """
    g_cdf = copula_grid(copula, N, use_cdf=True)
    g_pdf = copula_grid(copula, N, use_cdf=False, quad_order=10)
    inner = (slice(1, -1), slice(1, -1))
    a, b = g_cdf[inner], g_pdf[inner]
    err = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))
    ok = err < rtol
    if verbose:
        print(f"  {'OK  ' if ok else 'FAIL'} {_copula_label(copula):<34} "
              f"max interior rel. cell-mass mismatch {err:.2e}")
    return ok, err


def self_check(sigma=1.0):
    """Run the checks that matter before trusting a number out of this script.

    1. Every family's analytic CDF against its pdf quadrature.
    2. The independence -> Gaussian(0.9) fade length against 0.269632, the
       value independently verified (grid refinement + Richardson
       extrapolation + a spectral cross-check) when the generic solver was
       built. This ties the copula adapter -- not just the Part I engine --
       to an externally established number.
    3. The self-distance L(c, c) = 0.

    Part I has its own suite, which exercises it without any copula
    machinery: `python3 -m pytest test_w2_fade_path.py -v`.
    """
    print("CDF vs pdf-quadrature consistency (interior cells):")
    families = [
        load_target('IndependenceCopula', None, sigma),
        load_target('GaussianCopula', 0.7, sigma),
        load_target('ClaytonCopula', 2.0, sigma),
        load_target('GumbelCopula', 1.8, sigma),
        load_target('FrankCopula', 4.0, sigma),
        load_target('JoeCopula', 2.0, sigma),
        load_target('StudentTCopula', [0.5, 4.0], sigma),
        load_target('PlackettCopula', 0.3, sigma),   # theta < 1: negative dependence
        load_target('PlackettCopula', 6.0, sigma),
    ]
    families.append(MixtureCopula([families[2], families[4]], [0.4, 0.6]))
    # the mirrored CDF is assembled from the base one rather than transcribed,
    # so one representative per axis set is enough to pin the reflection algebra
    families += [MirroredCopula(families[2], 0),
                 MirroredCopula(families[3], (0, 1)),
                 MirroredCopula(families[6], 1)]
    # Khoudraji: asymmetric shapes, and a two-component form, since its CDF is
    # assembled from the components' rather than transcribed
    families += [KhoudrajiCopula(families[3], shape=(0.2, 0.8)),
                 KhoudrajiCopula(families[1], families[2], shape=(0.4, 0.7))]
    # Glued: one cut per coordinate. The cut positions are deliberately kept
    # on cell edges of the N=32 grid this check runs on (0.5 = 16/32,
    # 0.25 = 8/32) -- a glued density JUMPS across the seam, so in a cell the
    # seam runs through, the pdf-quadrature side of the comparison is simply
    # wrong (measured: up to 49% off at theta = 0.37) and would be testing
    # Gauss-Legendre rather than the CDF formula. That is exactly the failure
    # the CDF route exists to avoid; away from the seam the two agree to
    # ~2e-4, as below.
    families += [GluedCopula(families[2], families[3], 0.5, 0),
                 GluedCopula(families[1], families[5], 0.25, 1)]
    # Transposed: only worth checking on the two non-exchangeable
    # constructions, since the transpose of anything else is the identity.
    # The glued cut stays at 0.5 (a cell edge either side of the swap).
    families += [TransposedCopula(KhoudrajiCopula(families[3], shape=(0.2, 0.8))),
                 TransposedCopula(GluedCopula(families[2], families[3], 0.5, 0))]
    ok_all = all(check_cdf_consistency(c)[0] for c in families)

    print("\nMirroring identities (exact, not quadrature-limited):")
    # mirroring a Gaussian copula along one axis negates its correlation, so the
    # fade length between the two must vanish -- a check of the whole mirrored
    # path (CDF, cell masses, solver), not just of one density formula
    L0 = fade_distance(MirroredCopula(load_target('GaussianCopula', 0.7, sigma), 0),
                       load_target('GaussianCopula', -0.7, sigma), N=64, n_quad=24)["L"]
    print(f"  L(gaussian(0.7)@u0, gaussian(-0.7)) = {L0:.3e}")
    ok_all &= L0 < 1e-8

    # Plackett's parameter is a constant odds ratio, and reflecting one
    # coordinate inverts it, so mirroring along one axis is exactly Plackett(1/theta)
    Lp = fade_distance(MirroredCopula(load_target('PlackettCopula', 6.0, sigma), 0),
                       load_target('PlackettCopula', 1 / 6.0, sigma), N=64, n_quad=24)["L"]
    print(f"  L(plackett(6)@u0, plackett(1/6))    = {Lp:.3e}")
    ok_all &= Lp < 1e-8

    print("\nTranspose identities (exact, not quadrature-limited):")
    # every load_target family is exchangeable, so transposing one is a no-op
    gum = load_target('GumbelCopula', 1.8, sigma)
    Lt = fade_distance(TransposedCopula(gum), gum, N=64, n_quad=24)["L"]
    print(f"  L(gumbel(1.8)^T, gumbel(1.8))       = {Lt:.3e}")
    ok_all &= Lt < 1e-8

    # transposing a glued copula swaps which coordinate carries the seam:
    # T(glue(C1, C2, theta, 0)) = glue(T C1, T C2, theta, 1). Reads C over the
    # whole square, both strips, so it pins the swap against the gluing
    # algebra rather than against one density formula.
    kh = KhoudrajiCopula(gum, shape=(0.2, 0.8))
    cl = load_target('ClaytonCopula', 2.0, sigma)
    Lg1 = fade_distance(TransposedCopula(GluedCopula(kh, cl, 0.5, 0)),
                        GluedCopula(TransposedCopula(kh), TransposedCopula(cl),
                                    0.5, 1),
                        N=64, n_quad=24)["L"]
    print(f"  L(glue(K,clayton)^T, glue(K^T,..,1))= {Lg1:.3e}")
    ok_all &= Lg1 < 1e-8

    # mirror and transpose do not commute -- T . M_axes = M_(swapped axes) . T --
    # which is the relation the rotation recipes in TransposedCopula's
    # docstring rest on, so pin it on a non-exchangeable base
    Lc = fade_distance(TransposedCopula(MirroredCopula(kh, 0)),
                       MirroredCopula(TransposedCopula(kh), 1),
                       N=64, n_quad=24)["L"]
    print(f"  L(K@u0^T, K^T@u1)                   = {Lc:.3e}")
    ok_all &= Lc < 1e-8

    print("\nGluing identities (Siburg & Stoimenov 2008):")
    # Prop. 3.1: P glued to P is P again, and it is the only pair that gives
    # P back. Run through fade_distance, so this exercises the whole glued
    # path -- CDF, cell masses, solver -- not just one formula.
    ind = load_target('IndependenceCopula', None, sigma)
    Lg = fade_distance(GluedCopula(ind, ind, 0.37, 0), ind, N=64, n_quad=24)["L"]
    print(f"  L(glue(P, P, theta=0.37), P)        = {Lg:.3e}")
    ok_all &= Lg < 1e-8

    # Thm. 3.2: rho_C = theta^2 rho_1 + (1-theta)^2 rho_2, via Spearman's
    # rho = 12 * int_0^1 int_0^1 C(u,v) du dv - 3 (their eq. 13). This is the
    # sharpest cheap check on the construction: it reads C over the WHOLE
    # square, both strips at once, so a wrong scale factor or a swapped
    # coordinate shows up here even though the boundary values stay exact.
    nodes, wts = roots_legendre(256)
    nodes, wts = (nodes + 1) / 2, wts / 2
    Uq, Vq = np.meshgrid(nodes, nodes, indexing='ij')
    Wq = wts[:, None] * wts[None, :]

    def spearman_rho(c):
        return 12.0 * float(np.sum(copula_cdf_u(c)(Uq, Vq) * Wq)) - 3.0

    for c1, c2, th, idx in ((families[2], families[3], 0.4, 0),
                            (families[1], families[5], 0.65, 1),
                            (families[9], families[12], 0.3, 1)):
        glued = GluedCopula(c1, c2, th, idx)
        got = spearman_rho(glued)
        want = th ** 2 * spearman_rho(c1) + (1 - th) ** 2 * spearman_rho(c2)
        print(f"  rho(glue(.,., {th:g}, u{idx})) = {got:+.6f}   "
              f"Thm 3.2 predicts {want:+.6f}   diff {abs(got - want):.1e}")
        ok_all &= abs(got - want) < 1e-4

    print("\nRegression against the independently verified reference:")
    c0 = load_target('IndependenceCopula', None, sigma)
    c1 = load_target('GaussianCopula', 0.9, sigma)
    L = fade_distance(c0, c1, N=128, n_quad=48)["L"]
    reference = 0.269632
    rel = abs(L - reference) / reference
    print(f"  L(independence -> gaussian(0.9)) = {L:.6f}  "
          f"reference {reference}  rel. err {rel:.2e}")
    ok_all &= rel < 5e-3

    print("\nSelf-distance:")
    L0 = fade_distance(c1, c1, N=64, n_quad=24)["L"]
    print(f"  L(c, c) = {L0:.3e}")
    ok_all &= L0 < 1e-8

    print("\n" + ("all checks passed" if ok_all else "SOME CHECKS FAILED"))
    return ok_all


# ===========================================================================
# 10. Entry point
# ===========================================================================

# Deliberately NOT a general command line. There used to be an argparse layer
# with a `--copula0 "0.5*gumbel:1.5+0.5*frank:4"` mini-language; it was
# deleted because nothing but its own `main()` ever called it and it had
# already drifted out of sync with the library (it could not express a
# KhoudrajiCopula, so the module advertised an interface that could not reach
# part of itself). Anything it did is one line of real Python:
#
#     python3 -c "import distance_calculation_campaign.fade_distance as CF; \
#                 from distance_calculation_campaign.copula_families import load_target; \
#                 print(CF.fade_distance(load_target('ClaytonCopula', 2, 1.0), \
#                                        load_target('GumbelCopula', 1.8, 1.0))['L'])"
#
# which, unlike the parser, can express every construction copula_families
# supports. Running the module directly runs the validation suite, which is
# the one shell entry point that is actually referenced elsewhere (CLAUDE.md,
# and test_w2_fade_path.py's docstring).

if __name__ == '__main__':
    raise SystemExit(0 if self_check() else 1)