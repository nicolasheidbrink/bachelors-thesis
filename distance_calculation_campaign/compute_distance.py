'''
This script computes the copula-constrained Wasserstein distance between two copulas 
using a velocity field obtained by calling 
`copula_constrained_wasserstein_velocity_field.compute_copula_vector_field`. 

The code is taken from Grothe and Luo (github.com/naluo9991/Wasserstein-Transport-within-Copula).
'''

import numpy as np
import torch
from torchdiffeq import odeint

from distance_calculation_campaign.copula_constrained_wasserstein_velocity_field import (
    compute_copula_vector_field, device)


def run_transport(copula0, copula1):
   
    field = compute_copula_vector_field(copula0, copula1)

    npts_t = 50
    npts_grid = 100

    p = (torch.arange(npts_grid, dtype=torch.float64) + 0.5) / npts_grid  # percentile levels
    u_marg = p  # u-coordinate of each grid row

    eps = 1e-6
    n_fine = 4000
    v_fine = torch.linspace(eps, 1 - eps, n_fine, dtype=torch.float64)  # fine grid for integrating c0(u_i, ·)

    uu = u_marg.view(npts_grid, 1).expand(npts_grid, n_fine)
    vv = v_fine.view(1, n_fine).expand(npts_grid, n_fine)
    uv = torch.stack([uu, vv], dim=-1).to(device)  # [100, n_fine, 2]

    with torch.no_grad():
        c_vals = copula0.density_u(uv).cpu()  # [100, n_fine] == c0(u_i, v_fine_j)

    dv = v_fine[1:] - v_fine[:-1]
    increments = 0.5 * (c_vals[:, 1:] + c_vals[:, :-1]) * dv  # trapezoidal rule
    F = torch.cat([torch.zeros(npts_grid, 1, dtype=torch.float64), torch.cumsum(increments, dim=1)], dim=1)
    F = F / F[:, -1:]  # normalize (int_0^1 c0(u, v) dv == 1 exactly; this just corrects for the [0,eps]/[1-eps,1] truncation)

    v_grid = torch.from_numpy(
        np.stack([np.interp(p.numpy(), F[i].numpy(), v_fine.numpy()) for i in range(npts_grid)])
    )  # [100, 100]: v_grid[i, j] is the v at percentile p[j] conditional on u_marg[i]

    u_grid = u_marg.view(npts_grid, 1).expand(npts_grid, npts_grid)
    u0 = torch.stack([u_grid, v_grid], dim=-1).reshape(-1, 2).double().to(device)  # [10000, 2] percentile grid of c0

    t_grid = torch.linspace(0, 1, npts_t).to(device)

    # 3d tensor shaped [50, 10000, 2] with exact u coordinates of every particle at every time step:
    samples_u = odeint(field, u0, t=t_grid, method="dopri5", atol=1e-5, rtol=1e-5).cpu()

    W_value = torch.zeros(npts_t)
    for i in range(npts_t):
        w = field(t_grid[i].to(device), samples_u[i].to(device))
        W_value[i] = (w**2).sum(1).mean()  # squared speed, averaged across all grid particles (i.e. weighted by c, since grid cells are equal-probability under c0)


    return {
        "copula_constrained_wasserstein": float(torch.sqrt(W_value.mean())),
        "L2_fade_distance": field.L_fade,
        "iterations": field.iterations,
    }


if __name__ == "__main__":
    run_transport()
