'''
This script trains the neural networks used to estimate the velocity field. the code
is mostly taken from Grothe and Luo 
(github.com/naluo9991/Wasserstein-Transport-within-Copula), with adaptations outlined 
in Appendix A of the thesis. 
'''
import copy
import math
import types

import torch
import torch.nn as nn
from torch.func import vmap, jacrev

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# custom parametrized activation function = x * sigmoid(beta*x) with learnable parameter beta
class Swish(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5] * dim))

    def forward(self, x):
        return x * torch.sigmoid_(x * torch.nn.functional.softplus(self.beta))


# defines the MLP
class MLP(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = []
        for i in range(len(features)-2):
            layers.append(nn.Linear(features[i], features[i+1]))
            layers.append(Swish(features[i+1]).double().to(device))
        layers.append(nn.Linear(features[-2], features[-1]))
        self.params = nn.Sequential(*layers)

    def forward(self, x):
        return self.params(x)


# log rho neural network
class logrhonn_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_models = args.num_models # the amount of MLPs spaced across the timeline
        features = [args.ndim+1] + args.features + [1] # layer size for each MLP
        self.mlps = nn.ModuleList([MLP(features).double().to(device) for _ in range(args.num_models)]) # generates the MLPs

    # this function returns the estimated log rho_t(x) for the coordinates in `state` in a 1D tensor with the same amount of rows as state.
    # `state` is a matrix that has a column for time t and a column for x1 and a column for x2. the amount of rows is the batch size evaluated.
    def forward(self, state):
        logrhonns = [] # list of predictions from each of the MLPs
        for i in range(self.num_models):
            logrhonns.append(self.mlps[i](state))
        logrhonns = torch.stack(logrhonns) # creates a pytorch tensor out of the indiviual predictions
        T = state[...,0].unsqueeze(-1) # saves the times t in the right shape for later
        time_centers = torch.linspace(0, 1, self.num_models).to(state) # creates evenly spread anchor points around time interval [0,1], one anchor point per `num_models`
        time_weights = torch.exp(-((time_centers - T) ** 2) * (self.num_models ** 2)) # weights calculated for each of the times from T using RBF
        if state.ndim >1:
            return torch.einsum("ijk,ji->jk", logrhonns, time_weights)
        elif state.ndim == 1:
            return (logrhonns * time_weights.unsqueeze(-1)).sum(0)

    # returns analytically calculatedpartial derivatives of the output log-density wrt t and all x
    def d_logrhonn_temp(self, state):
        return jacrev(self.forward)(state).squeeze()

    # does multiple `d_logrhonn_temp` in parallel
    def d_logrhonn(self, state):
        return vmap(self.d_logrhonn_temp)(state)


class vector_model(nn.Module):
    # initializes an ensemble of MLPs to predict the 2D field for velocity v_t
    def __init__(self, args):
        super().__init__()
        self.num_models = args.num_models
        features = [args.ndim+1] + args.features + [2]
        self.mlps = nn.ModuleList([MLP(features).double().to(device) for _ in range(args.num_models)])

    # computes the continuous 2d vector field for given set of particles
    # outputs a tensor of shape [N, 2] with the vector field values for each of the N input particles from `state`
    def forward(self, state):
        vectors = []
        for i in range(self.num_models):
            vectors.append(self.mlps[i](state))
        vectors = torch.stack(vectors)
        T = state[...,0].unsqueeze(-1)
        time_centers = torch.linspace(0, 1, self.num_models).to(state)
        time_weights = torch.exp(-((time_centers - T) ** 2) * (self.num_models ** 2))
        if state.ndim >1:
            return torch.einsum("ijk,ji->jk", vectors, time_weights)
        elif state.ndim == 1:
            return (vectors * time_weights.unsqueeze(-1)).sum(0)

    def dx_vector_temp(self, state):
        return torch.trace(jacrev(self.forward)(state)[...,1:])

    def dx_vector(self, state):
        return vmap(self.dx_vector_temp)(state)


# learnes the integral of rho (technically its log) to normalize it to a probability measure
class constant_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        features = [1] + args.features + [1]
        self.mlps = MLP(features).double().to(device)

    def forward(self, t):
        return self.mlps(t)


# Initializes the three sub-networks and binds them into a single unified PyTorch module so the optimizer can update all their weights concurrently
class joint_wrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.logrhonn_model = logrhonn_model(args)
        self.vector_model = vector_model(args)
        self.constant_model = constant_model(args)

    def forward(self, state):
        logrhonn = self.logrhonn_model(state) # calling the `__call__` method of the object initialized in constructor, which then calls the `forward` method
        vec = self.vector_model(state)
        t = state[...,0:1]
        C = self.constant_model(t)
        return logrhonn, vec, C

    # returns the gradient terms required to evaluate the continuity equation
    def d_model(self, state):
        dlogrhonn = self.logrhonn_model.d_logrhonn(state)
        dvec = self.vector_model.dx_vector(state)
        dtlogrhonn, dxlogrhonn = dlogrhonn[...,0], dlogrhonn[...,1:]
        return dtlogrhonn, dxlogrhonn, dvec


class Workspace:
    def __init__(self, args, target0, target1):
        self.args = args # hyperparams for learning rate, nr of iterations, network sizes
        self.target0 = target0.double().to(device) # starting copula distribution
        self.target1 = target1.double().to(device) # ending copula distribution
        self.iter = 0
        self.module = joint_wrapper(args).double().to(device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=args.lr)
        self.L_fade = self._fade_upper_bound()

    def _fade_upper_bound(self):
        try:
            from distance_calculation_campaign.fade_distance import fade_distance
            res = fade_distance(self.target0, self.target1, space="u")
            L = float(res['L'])
        except Exception as exc:
            if self.args.verbose:
                print(f'fade-path upper bound unavailable ({type(exc).__name__}: {exc}) -- '
                      f'continuing without it')
            return None
        if self.args.verbose:
            print(f'fade-path upper bound: L_fade={L:.4f} '
                  f'(compare against sqrt(loss_energy))')
        return L

    def train(self):
        window = self.args.window                # how many iterations of history a plateau check looks back over
        energy_plateau_rel_tol = self.args.energy_plateau_rel_tol # loss_energy counts as "stopped shrinking" if it improves less than x% over a window
        penalty_weight_increment = self.args.penalty_weight_increment # amount penalty_weight is raised by each time it's raised

        energy_history = []
        con_history = []
        con_tol = self.args.con_tol
        self.checkpoints = {}  # {iteration: state_dict} snapshots, kept in memory only

        # grid resolution and domain bounds are fixed hyperparameters; per-axis spacing
        # is used below as the half-width of each iteration's random jitter
        n_point = self.args.n_point # defines resolution of time and space
        dt = 1.0 / (n_point - 1) if n_point > 1 else 0.0
        dx = 2.0 / (n_point - 1) if n_point > 1 else 0.0
        t_anchor = torch.linspace(0,1,n_point).double().to(device)
        x_anchor = torch.linspace(-1,1,n_point).double().to(device)

        state = None

        try:
            while self.iter <= self.args.num_iterations:
                # The grid and the endpoint densities on it don't depend on the model's
                # weights, so on a fixed grid this block runs once and everything it
                # binds is reused by every later iteration. With args.jitter_grid it
                # instead reruns each iteration, perturbing every anchor by up to half
                # a grid cell around its regular linspace position, so training never
                # sees the exact same finite set of points (which risks the model
                # overfitting to those points rather than the underlying continuous
                # PDE). Either way the (n_point, n_point, n_point) cartesian-product
                # SHAPE is kept, since the marginal-density/flux computations below
                # (f1, f2, fv_1, fv_2) rely on it via reshape+mean.
                if state is None or self.args.jitter_grid:
                    if self.args.jitter_grid:
                        t = (t_anchor + (torch.rand(n_point).double().to(device)-0.5)*dt).clamp(0,1)
                        x1 = (x_anchor + (torch.rand(n_point).double().to(device)-0.5)*dx).clamp(-1,1)
                        x2 = (x_anchor + (torch.rand(n_point).double().to(device)-0.5)*dx).clamp(-1,1)
                    else:
                        t, x1, x2 = t_anchor, x_anchor, x_anchor
                    state = torch.cartesian_prod(t, x1, x2)
                    t_list = state[...,0]
                    x_list = state[...,1:]
                    logf0 = self.target0.log_density(x_list) # vector of the log of the pdf of the distribution target0 of each x from the grid
                    logf1 = self.target1.log_density(x_list) # vector of the log of the pdf of the distribution target1 of each x from the grid
                    dx_logden0 = self.target0.dx_logden(x_list)
                    dx_logden1 = self.target1.dx_logden(x_list)

                logrhonn, vec, C = self.module(state) # calculates unnormalized density, velocity, and normalization constant
                logrhonn = logrhonn.squeeze()
                C = C.squeeze()
                dtlogrhonn, dxlogrhonn, dvec = self.module.d_model(state) # gets temporal and spacial gradients

                logrho = (1-t_list)*logf0 + t_list*logf1 + t_list*(1-t_list)*logrhonn # the real rho is calculated as the geometric mean of the start and end distribution
                rho = torch.exp(logrho)                                               # ... multiplied by rhonn^(t(1-t)). If logrhonn > 0, then probability mass increases at that point
                dt_logrho = -logf0 + logf1 + (1-2*t_list)*logrhonn + t_list*(1-t_list)*dtlogrhonn
                dx_logrho = (1-t_list).unsqueeze(-1)*dx_logden0 + t_list.unsqueeze(-1)*dx_logden1 \
                           + (t_list*(1-t_list)).unsqueeze(-1)*dxlogrhonn
                rho_reshape = rho.reshape((n_point,n_point,n_point)) # creates 3d grid
                # the following line normalizes the spacial density at every t so that it integrates to 1
                # the integral is meeanheight * basearea
                f_reshape = rho_reshape / (rho_reshape.mean([1,2]) * 2**2).unsqueeze(-1).unsqueeze(-1).repeat(1, n_point, n_point)
                f = f_reshape.reshape(rho.shape) # puts the 3d grid back into 1d vector shape
                # the following line calculates the loss from violating the continuity equation. everything up to and including dvec is just the continuity equation (eq. 4.3).
                # multiply by f to penalize errors in dense areas more highly. taking the mean and multiplying by base area is integrating over x
                loss_con = torch.abs((dt_logrho + (dx_logrho * vec).sum(1) + dvec + C)*f).mean() * 2**2

                f1 = f_reshape.mean(2) * 2 # calculates marginal densities
                f2 = f_reshape.mean(1) * 2 # calculates marginal densities
                f1 = f1.unsqueeze(-1).repeat(1, 1, n_point).reshape(f.unsqueeze(-1).shape)
                f2 = f2.unsqueeze(1).repeat(1, n_point, 1).reshape(f.unsqueeze(-1).shape)
                fv = f.unsqueeze(-1) * vec # momentum field (aka flux)
                # next line: calculate marginal fluxes by integrating momentum field along one spatial axis (h_1 \bar v_1 and h_2 \bar v_2)
                fv_1 = fv[...,0].reshape((n_point,n_point,n_point)).mean(2).unsqueeze(-1).repeat(1, 1, n_point).reshape(f.unsqueeze(-1).shape) * 2
                fv_2 = fv[...,1].reshape((n_point,n_point,n_point)).mean(1).unsqueeze(1).repeat(1, n_point, 1).reshape(f.unsqueeze(-1).shape) * 2
                f_mean = torch.cat([f1,f2], dim=1)
                fv_mean = torch.cat([fv_1,fv_2], dim=1)
                w = f_mean*vec - fv_mean # copula velocity field without marginal flux
                loss_energy = ((w**2).sum(1)*f).mean() * 2**2 # penalizes transport cost

                # MLP loss function
                loss = loss_energy + self.args.penalty_weight*loss_con

                self.optimizer.zero_grad() # clears the gradients from the previous iteration
                loss.backward() # calculates gradient of the loss wrt every learnable weight
                grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1) # scales gradients down if their norm > 1
                if ~torch.isfinite(grad_norm): # if NaN
                    raise RuntimeError(
                        f"Gradient norm is {grad_norm} while loss is {loss.item()}"
                    )
                self.optimizer.step() # executes weight update

                energy_history.append(loss_energy.item())
                con_history.append(loss_con.item())

                # every `window` iterations, check whether loss_energy has plateaued under the
                # current (fixed) penalty_weight; if so, either stop (constraint already small
                # too) or grow the penalty and keep going, judging the next plateau afresh
                if len(energy_history) == window:
                    # in-memory snapshot, so an interrupted (or completed) run can still be
                    # inspected as it looked at an earlier iteration -- see get_checkpoints().
                    self.checkpoints[self.iter + 1] = {k: v.detach().clone() for k, v in self.module.state_dict().items()}

                    half = window // 2
                    prior_energy_mean = sum(energy_history[:half]) / half
                    recent_energy_mean = sum(energy_history[half:]) / (window - half)
                    recent_con_mean = sum(con_history[half:]) / (window - half)
                    relative_improvement = (prior_energy_mean - recent_energy_mean) / abs(prior_energy_mean)
                    plateaued = relative_improvement < energy_plateau_rel_tol
                    energy_history = []
                    con_history = []

                    if plateaued and recent_con_mean < con_tol:
                        if self.iter <= 10000:
                            if self.args.verbose:
                                print(f'iteration {self.iter}/{self.args.num_iterations}: little to no improvement. '
                                      f'loss_energy relative improvement is {relative_improvement:.4f} and loss_con is {recent_con_mean:.3e} '
                                      'Not yet stopping because not enough iterations have passed yet. ')
                        elif (self.L_fade is None or math.sqrt(recent_energy_mean) < self.L_fade + self.args.upper_bound_tol):
                            if self.args.verbose:
                                print(f'iteration {self.iter}/{self.args.num_iterations}: stopping early -- '
                                    f'loss_energy plateaued (rel. improvement {relative_improvement:.4f} < cutoff {energy_plateau_rel_tol:.4f}) '
                                    f'and loss_con={recent_con_mean:.3e} < tol={con_tol:.3e}')
                            break
                        else:
                            if self.args.verbose:
                                print(f'iteration {self.iter}/{self.args.num_iterations}: loss_energy plateaued '
                                      f'(rel. improvement {relative_improvement:.4f} < cutoff {energy_plateau_rel_tol:.4f}) '
                                      f'and loss_con={recent_con_mean:.3e} < tol={con_tol:.3e}, but '
                                      f'sqrt(loss_energy)={math.sqrt(recent_energy_mean):.4f} >= '
                                      f'L_fade+tol={self.L_fade + self.args.upper_bound_tol:.4f} -- not converged, '
                                      f'the training-free fade path is still cheaper; continuing')
                    elif plateaued and recent_con_mean >= con_tol:
                        self.args.penalty_weight = self.args.penalty_weight + penalty_weight_increment
                        if self.args.verbose:
                            print(f'iteration {self.iter}/{self.args.num_iterations}: loss_energy plateaued '
                                  f'(rel. improvement {relative_improvement:.4f} < cutoff {energy_plateau_rel_tol:.4f}) but '
                                  f'loss_con={recent_con_mean:.3e} >= tol={con_tol:.3e} -- '
                                  f'raising penalty_weight to {self.args.penalty_weight:.3g}')
                    else:
                        if self.args.verbose:
                            print(f'iteration {self.iter}/{self.args.num_iterations}: loss_energy still improving '
                                  f'(rel. improvement {relative_improvement:.4f} >= cutoff {energy_plateau_rel_tol:.4f})')

                if self.args.verbose and self.iter % 250 == 0:
                    print(f'iteration {self.iter}/{self.args.num_iterations}: loss={loss.item():.3f} (loss_energy={loss_energy.item():.3f}, loss_con={loss_con.item():.3f}, penalty_weight={self.args.penalty_weight:.3g})')
                self.iter += 1
        except KeyboardInterrupt:
            if self.args.verbose:
                print(f'iteration {self.iter}/{self.args.num_iterations}: training interrupted -- returning model as currently trained')



    def get_model(self):
        return self.module

    def get_checkpoints(self):
        return self.checkpoints


class CopulaVelocityField:
    """Callable w(t, u) recovered from a trained model via the Sklar-type
    decomposition (ported from the notebook's `wu_func`). Pass directly as
    `torchdiffeq.odeint`'s first argument, or call it directly as
    `field(t, u)`.
    """

    def __init__(self, module, copula0, copula1, device, *, grid_res=500, checkpoints=None,
                 L_fade=None, iterations=None):
        self.module = module
        self.copula0 = copula0
        self.copula1 = copula1
        self.device = device
        self.grid_res = grid_res
        self.checkpoints = checkpoints or {}  # {iteration: state_dict}, see Workspace.get_checkpoints()
        # Two facts about the run that produced these weights, carried along so a
        # caller holding only the field can report them without re-deriving:
        self.L_fade = L_fade          # Workspace.L_fade, the fade-path upper bound (u-space
                                      # length); None if that solve was unavailable
        self.iterations = iterations  # the iteration Workspace.train()'s loop exited at --
                                      # num_iterations+1 if it ran to completion, lower if it
                                      # stopped early on a plateau or was interrupted

    def field_at_checkpoint(self, iteration):
        """Returns a new CopulaVelocityField using the model weights as they
        were at `iteration` (one of the keys of `.checkpoints`) instead of
        the final trained weights -- lets you see what the transport would
        have looked like had training been interrupted at that point."""
        module = copy.deepcopy(self.module)
        module.load_state_dict(self.checkpoints[iteration])
        # L_fade depends only on the endpoints, so it carries over unchanged; the
        # snapshot's own iteration replaces the full run's count.
        return CopulaVelocityField(module, self.copula0, self.copula1, self.device,
                                   grid_res=self.grid_res, L_fade=self.L_fade,
                                   iterations=iteration)

    @torch.no_grad() # disables PyTorch's autograd engine
    def __call__(self, t, u):
        """
        GPU-safe, vectorized velocity function for odeint.

        Args:
            t: scalar time
            u: shape (batch, 2), current state (u1, u2)

        Returns:
            w: shape (batch, 2), velocity
        """
        module, target0, target1, device = self.module, self.copula0, self.copula1, self.device
        x1_grid = torch.linspace(-1, 1, self.grid_res).double().to(device) # high resolution spacial grid
        x2_grid = torch.linspace(-1, 1, self.grid_res).double().to(device)
        u1, u2 = u[..., 0], u[..., 1]
        eps = 1e-8
        u1 = torch.clamp(u1, 0.0, 1.0 - eps) # enforces boundaries of copula domain
        u2 = torch.clamp(u2, 0.0, 1.0 - eps)

        X1, X2 = torch.meshgrid(x1_grid, x2_grid, indexing="ij")
        X1f = X1.reshape(-1) # flattens 2d grid into 1d vector
        X2f = X2.reshape(-1)
        t_list = torch.full_like(X1f, t) # vector with same length as x2f and x2f that just contains t

        state = torch.stack([t_list, X1f, X2f], dim=1) # combines time and space to create input matrix
        logrhonn, vec, C = module(state) # predicts unnormalized density, velocity field, and normalization constant for time t
        logrhonn = logrhonn.squeeze()
        C = C.squeeze()

        t_list = state[...,0]
        x_list = state[...,1:]
        logf0 = target0.log_density(x_list)
        logf1 = target1.log_density(x_list)
        logrho = (1-t_list)*logf0 + t_list*logf1 + t_list*(1-t_list)*logrhonn
        rho = torch.exp(logrho)
        f = rho / (rho.mean()*2**2)
        pdf = f.reshape(len(x1_grid), len(x2_grid))
        flux = (f.unsqueeze(-1) * vec).reshape(len(x1_grid), len(x2_grid), 2) # [grid_res, grid_res, 2] 3d grid storing directional flux at each point

        pdf_x1 = torch.trapz(pdf, x2_grid, dim=1)  # marginal along x1 using trapz instead of mean for more accuracy
        cdf_x1 = torch.cumsum(pdf_x1, dim=0) # CDF
        cdf_x1 = cdf_x1/cdf_x1[-1] # divides CDF by final value to make sure it really ends up being 1
        cdf_x1 = torch.clamp(cdf_x1, min=eps, max=1.0)

        pdf_x2 = torch.trapz(pdf, x1_grid, dim=0)  # marginal along x2
        cdf_x2 = torch.cumsum(pdf_x2, dim=0)
        cdf_x2 = cdf_x2/cdf_x2[-1]
        cdf_x2 = torch.clamp(cdf_x2, min=eps, max=1.0)

        idx1 = torch.searchsorted(cdf_x1, u1) # finds grid index where u_1 falls within the CDF
        idx1 = torch.clamp(idx1, 0, len(x1_grid) - 1)
        x1 = x1_grid[idx1] # maps that index back to physical coordinate grid, translating the particle's copula location to its physical location in x domain

        idx2 = torch.searchsorted(cdf_x2, u2)
        idx2 = torch.clamp(idx2, 0, len(x2_grid) - 1)
        x2 = x2_grid[idx2]

        fv1 = torch.trapz(flux[..., 0], x2_grid, dim=1)[idx1]  # shape (batch,) ; finds marginal momentum h_i \bar v_i
        fv2 = torch.trapz(flux[..., 1], x1_grid, dim=0)[idx2]  # shape (batch,)

        t_list = torch.full_like(x1, t)
        y_point = torch.stack([t_list, x1, x2], dim=1)
        _, v, _ = module(y_point) # gets exact joint velocity vectors at (t, x1, x2)

        w = torch.zeros_like(v)
        w[:, 0] = pdf_x1[idx1] * v[:, 0] - fv1 # calculates sklar-type decomposition w = h*(v - \bar v)
        w[:, 1] = pdf_x2[idx2] * v[:, 1] - fv2
        return w


def compute_copula_vector_field(
    copula0,
    copula1,
    *,
    features=[256, 256, 256],
    num_models=4,
    lr=1e-3,
    num_iterations=150000,
    penalty_weight=5,
    window=10000,                # how many iterations of history a plateau check looks back over
    energy_plateau_rel_tol=0.008, # loss_energy counts as "stopped shrinking" if it improves less than x% over a window
    penalty_weight_increment=1,  # amount penalty_weight is raised by each time it's raised
    con_tol=0.05,
    upper_bound_tol=0.02,        # slack allowed when checking a trained action against Workspace.L_fade
    n_point=16,
    jitter_grid=True,            # rebuild the collocation grid with fresh jitter every iteration
    grid_res=500,
    verbose=True,
):
    """Train a copula dynamical optimal transport model between copula0 and
    copula1 and return the learned copula-domain velocity field.

    copula0, copula1: `load_target` instances that must share the same
        `.sigma` -- both endpoints must use the same fixed marginal, per the
        paper's Sklar-type decomposition (Theorem 3.1).

    Returns a `CopulaVelocityField`, callable as `field(t, u) -> w`, usable
    directly as the right-hand side of `torchdiffeq.odeint`.

    upper_bound_tol: slack allowed when checking a trained action against
        `Workspace.L_fade`, the training-free upper bound the workspace computes
        for every run (see `Workspace._fade_upper_bound`).

    jitter_grid: if True (default), the (n_point, n_point, n_point) collocation
        grid is rebuilt every iteration with each axis's anchors perturbed by up
        to half a grid cell, so training never revisits the exact same finite
        set of points. If False, one fixed regular grid is used for the whole
        run and the endpoint densities/gradients on it are computed once up
        front -- cheaper per iteration (noticeably so for families like
        Student-t whose density goes through a scipy round-trip), at the risk of
        the model overfitting to that particular point set.

    Defaults mirror the reference notebook's example run. Note the spatial
    domain is hardcoded to [-1, 1]^2, so choose sigma so that the marginal's
    probability mass is well contained in that range (sigma=0.26 in the
    reference experiments).
    """
    if float(copula0.sigma) != float(copula1.sigma):
        raise ValueError("copula0 and copula1 must share the same sigma (fixed marginal).")

    args = types.SimpleNamespace(
        features=list(features),   # must be a real list: the MLP builders below do [x] + args.features + [y]
        num_models=num_models,
        ndim=2,                    # hardcoded: vector_model/CopulaVelocityField are 2D-only throughout
        lr=lr,
        num_iterations=num_iterations,
        penalty_weight=penalty_weight,
        window=window,
        energy_plateau_rel_tol=energy_plateau_rel_tol,
        penalty_weight_increment=penalty_weight_increment,
        con_tol=con_tol,
        upper_bound_tol=upper_bound_tol,
        n_point=n_point,
        jitter_grid=jitter_grid,
        verbose=verbose,
        
    )
    workspace = Workspace(args, copula0, copula1)
    workspace.train()
    return CopulaVelocityField(
        workspace.get_model(), copula0, copula1, device,
        grid_res=grid_res, checkpoints=workspace.get_checkpoints(),
        L_fade=workspace.L_fade, iterations=workspace.iter,
    )
