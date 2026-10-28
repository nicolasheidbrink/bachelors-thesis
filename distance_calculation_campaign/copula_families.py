'''
This script contains formulas for each of the copula families used.

Some of the families were taken from Grothe and Luo 
    (github.com/naluo9991/Wasserstein-Transport-within-Copula).
Other families were implemented with the help of Claude Code.
'''

import math

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev
from scipy.stats import norm
from scipy.stats import t as _t_dist
from scipy.special import gamma, beta, gammaln, ndtr, roots_legendre
from statsmodels.distributions.copula.api import (
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    GumbelCopula,
    FrankCopula,
    IndependenceCopula,
)

normal = torch.distributions.Normal(0.0, 1.0)


def _rvs_sibuya(alpha, size):
    '''
    This method is needed for random sampling from Joe copulas.
    Other families have samplers `rvs()` from statsmodels, but none is implemented for Joe Copulas.
    Sampling from the Sibuya(alpha) distribution is a step to generate random variables:

        x = np.random.random((n_samples, 2))
        v = _rvs_sibuya(1 / theta, n_samples)
        s = -np.log(x) / v
        sample = 1 - (-np.expm1(-s)) ** (1 / theta)

    It was generated with Claude Code.
    '''
    u = np.random.random(size)
    v = np.ones(size)
    if alpha < 1:
        mask = u > alpha
        um = u[mask]
        ginv = ((1 - um) * gamma(1 - alpha)) ** (-1 / alpha)
        floor_ginv = np.floor(ginv)
        accept_ceil = (1 - 1 / (floor_ginv * beta(floor_ginv, 1 - alpha))) < um
        v[mask] = np.where(accept_ceil, np.ceil(ginv), floor_ginv)
    return v


class _StudentTICDF(torch.autograd.Function):
    '''
    This class is used to calculate the inverse CDF of a Student-t distribution using PyTorch autograd.
    torch.distributions.StudentT does not have an icdf, like torch.distributions.Normal does.
    
    It was generated with Claude Code.
    '''

    @staticmethod
    def forward(p, nu):
        nu_f = float(nu.item() if isinstance(nu, torch.Tensor) else nu)
        p_np = p.detach().cpu().numpy()
        # np.asarray: scipy's ppf demotes a 0-d input to a bare numpy scalar,
        # which torch.from_numpy rejects -- this is the unbatched case, i.e. a
        # single point x of shape (2,) rather than a batch of shape (N, 2)
        x_np = np.asarray(_t_dist.ppf(p_np, df=nu_f))
        return torch.from_numpy(x_np).to(p)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, nu = inputs
        ctx.nu = float(nu.item() if isinstance(nu, torch.Tensor) else nu)
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        log_pdf = torch.distributions.StudentT(torch.tensor(ctx.nu).to(x)).log_prob(x)
        grad_p = grad_output * torch.exp(-log_pdf)
        return grad_p, None


def _student_t_icdf(p, nu):
    return _StudentTICDF.apply(p, nu)


# u is clipped this far from {0, 1} before an inverse CDF or a log sees it.
_UEPS = 1e-12


class _StudentTCDF(torch.autograd.Function):
    '''
    Again, PyTorch does not provide the CDF for Student-t distributions.

    This was also generated with Claude Code.
    '''

    @staticmethod
    def forward(x, nu):
        nu_f = float(nu.item() if isinstance(nu, torch.Tensor) else nu)
        y = np.asarray(_t_dist.cdf(x.detach().cpu().numpy(), df=nu_f))
        return torch.from_numpy(y).to(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, nu = inputs
        ctx.nu = float(nu.item() if isinstance(nu, torch.Tensor) else nu)
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        log_pdf = torch.distributions.StudentT(torch.tensor(ctx.nu).to(x)).log_prob(x)
        return grad_output * torch.exp(log_pdf), None


def _student_t_cdf(x, nu):
    return _StudentTCDF.apply(x, nu)


def bivariate_normal_cdf(a, b, rho, n_nodes=96, lower=-9.0):
    '''
    P(X <= a, Y <= b) for a standard bivariate normal with correlation rho.

    This method was generated with Claude Code.
    '''
    a = np.clip(np.asarray(a, dtype=float), lower, -lower)
    b = np.asarray(b, dtype=float)
    s = np.sqrt(max(1.0 - float(rho) ** 2, 1e-300))

    gx, gw = roots_legendre(n_nodes)
    half = 0.5 * (a - lower)
    mid = 0.5 * (a + lower)
    X = mid[..., None] + half[..., None] * gx                      # (..., Q)
    integrand = (np.exp(-0.5 * X ** 2) / np.sqrt(2 * np.pi)
                 * ndtr((b[..., None] - float(rho) * X) / s))
    return np.clip(half * (integrand * gw).sum(-1), 0.0, 1.0)


def bivariate_t_cdf(a, b, rho, nu, n_nodes=200):
    '''
    P(S <= a, T <= b) for a standard bivariate t (correlation rho, df nu).

    This method was generated with Claude Code.
    '''
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rho, nu = float(rho), float(nu)
    K = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi)

    upper = np.arctan(a / np.sqrt(nu))
    lower = -np.pi / 2
    gx, gw = roots_legendre(n_nodes)
    half = 0.5 * (upper - lower)
    mid = 0.5 * (upper + lower)
    phi = mid[..., None] + half[..., None] * gx                    # (..., Q)
    X = np.sqrt(nu) * np.tan(phi)
    arg = ((b[..., None] - rho * X)
           / np.sqrt(max(1.0 - rho ** 2, 1e-300) * (nu + X ** 2) / (nu + 1)))
    integrand = K * np.cos(phi) ** (nu - 1) * _t_dist.cdf(arg, df=nu + 1)
    return np.clip(half * (integrand * gw).sum(-1), 0.0, 1.0)


class _BivariateNormalCDF(torch.autograd.Function):
    '''
    This class does the same as the method above, except it uses PyTorch, meaning the derivative 
    can be determined more easily. That is necessary for the method `cdf_u` and Khoudraji devices, 
    which use copulas' CDFs

    It was generated with Claude Code.
    '''

    @staticmethod
    def forward(a, b, rho):
        out = bivariate_normal_cdf(a.detach().cpu().numpy(),
                                   b.detach().cpu().numpy(), float(rho))
        return torch.from_numpy(np.asarray(out)).to(a)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b, rho = inputs
        ctx.rho = float(rho)
        ctx.save_for_backward(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        rho = ctx.rho
        s = math.sqrt(max(1.0 - rho ** 2, 1e-300))
        norm_pdf = lambda z: torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        da = norm_pdf(a) * torch.special.ndtr((b - rho * a) / s)
        db = norm_pdf(b) * torch.special.ndtr((a - rho * b) / s)
        return g * da, g * db, None


class _BivariateTCDF(torch.autograd.Function):
    '''
    This class for Student-t distributions exists for the same purpose as the above class for normal distributions
    '''

    @staticmethod
    def forward(a, b, rho, nu):
        out = bivariate_t_cdf(a.detach().cpu().numpy(),
                              b.detach().cpu().numpy(), float(rho), float(nu))
        return torch.from_numpy(np.asarray(out)).to(a)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b, rho, nu = inputs
        ctx.rho, ctx.nu = float(rho), float(nu)
        ctx.save_for_backward(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        rho, nu = ctx.rho, ctx.nu
        one_m = max(1.0 - rho ** 2, 1e-300)
        t_pdf = lambda z: torch.exp(torch.distributions.StudentT(
            torch.tensor(nu).to(z)).log_prob(z))
        cond = lambda z, w: _student_t_cdf(
            (w - rho * z) / torch.sqrt(one_m * (nu + z ** 2) / (nu + 1)), nu + 1)
        return g * t_pdf(a) * cond(a, b), g * t_pdf(b) * cond(b, a), None, None


def _bivariate_normal_cdf_t(a, b, rho):
    return _BivariateNormalCDF.apply(a, b, rho)


def _bivariate_t_cdf_t(a, b, rho, nu):
    return _BivariateTCDF.apply(a, b, rho, nu)


class load_target(nn.Module):
    '''
    This class contains the methods for regular copulas (not linear combinations or Khoudraji devices).
    `target` is the copula family, `theta` contains its parameters, and `Sigma` is the standard deviation of 
    the two underlying normal distributions connected by the copula.
    '''
    def __init__(self, target, theta, Sigma):
        super().__init__()
        # For Packett copulas with negative correlations, use the inverse theta and mirror along axis 0
        if target == 'PlackettCopula' and theta is not None and float(theta) <= 0:
            raise ValueError("PlackettCopula requires theta > 0 "
                             "(theta -> 1 is the independence limit)")
        self.theta = torch.tensor(theta) if theta is not None else None
        self.target = target
        self.sigma = Sigma

    # Shared algebra for the Plackett family. Its natural parameter is
    # eta = theta - 1 (eta = 0 is independence, eta < 0 negative dependence),
    # and everything -- CDF, density, h-function, gradient -- is built from
    # the same three quantities, so they are computed in one place:
    #     S = 1 + eta(u+v),  D = S^2 - 4 theta eta u v,  N = 1 + eta(u+v-2uv)
    # D is the discriminant of the quadratic that defines the copula; it is
    # positive on the open square, and clamped only to keep sqrt/log finite
    # exactly at the boundary.
    def _plackett_parts(self, u, v):
        th = self.theta.to(u.dtype)
        eta = th - 1
        S = 1 + eta * (u + v)
        D = (S ** 2 - 4 * th * eta * u * v).clamp_min(_UEPS)
        N = 1 + eta * (u + v - 2 * u * v)
        return th, eta, S, D, N

    # computes denstity f(x1, x2) of the original joint random variable (R^2 -> R^+)
    # Other than GaussianCopula and IndependenceCopula, Sklar's theorem is used: f(x1, x2) = c(F1(x1), F2(x2)) * f1(x1) * f2(x2)
    def density_x(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        x,y = x[...,0], x[...,1]
        if self.target == 'GaussianCopula':
            output = 1/(2*torch.pi*torch.sqrt(1-self.theta**2)*self.sigma**2)*torch.exp((2*self.theta*x*y-(x**2+y**2))/(2*self.sigma**2*(1-self.theta**2)))
            return output
        elif self.target == 'ClaytonCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            output = (1+self.theta)*(u*v)**(-self.theta-1)*(u**(-self.theta)+v**(-self.theta)-1)**(-1/self.theta-2)
            output = output / (2*torch.pi*self.sigma**2) * torch.exp(-(x**2+y**2)/(2*self.sigma**2))
            return output
        elif self.target == 'GumbelCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            logu = -torch.log(u)
            logv = -torch.log(v)
            tmp = (logu)**self.theta + (logv)**self.theta
            output = 1/(u*v)*(logu * logv)**(self.theta-1)*torch.exp(-tmp**(1/self.theta))*tmp**(1/self.theta-2)*(tmp**(1/self.theta)+self.theta-1)
            output = output / (2*torch.pi*self.sigma**2) * torch.exp(-(x**2+y**2)/(2*self.sigma**2))
            return output
        elif self.target == 'FrankCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            output = self.theta*(1-torch.exp(-self.theta))*torch.exp(-self.theta*(u+v)) / (torch.exp(-self.theta*u)+torch.exp(-self.theta*v)-
                                                                            torch.exp(-self.theta*(u+v))-torch.exp(-self.theta))**2
            output = output / (2*torch.pi*self.sigma**2) * torch.exp(-(x**2+y**2)/(2*self.sigma**2))
            return output
        elif self.target == 'JoeCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            tempg = (1-u)**(self.theta) + (1-v)**(self.theta) - (1-u)**(self.theta)*(1-v)**(self.theta)
            output = (1-u)**(self.theta-1) * (1-v)**(self.theta-1) * tempg**(1/self.theta-2)*(self.theta-1+tempg)
            output = output / (2*torch.pi*self.sigma**2) * torch.exp(-(x**2+y**2)/(2*self.sigma**2))
            return output
        elif self.target == 'StudentTCopula':
            rho, nu = self.theta[0], self.theta[1]
            u,v = normal2.cdf(x), normal2.cdf(y)
            s, w = _student_t_icdf(u, nu), _student_t_icdf(v, nu)
            A = 1 + (s**2 + w**2 - 2*rho*s*w) / (nu*(1-rho**2))
            B1, B2 = 1 + s**2/nu, 1 + w**2/nu
            K = torch.exp(torch.lgamma((nu+2)/2) + torch.lgamma(nu/2) - 2*torch.lgamma((nu+1)/2))
            output = K / torch.sqrt(1-rho**2) * A**(-(nu+2)/2) * B1**((nu+1)/2) * B2**((nu+1)/2)
            output = output / (2*torch.pi*self.sigma**2) * torch.exp(-(x**2+y**2)/(2*self.sigma**2))
            return output
        elif self.target == 'PlackettCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            th, eta, S, D, N = self._plackett_parts(u, v)
            output = th * N / D**1.5
            output = output / (2*torch.pi*self.sigma**2) * torch.exp(-(x**2+y**2)/(2*self.sigma**2))
            return output
        elif self.target == 'IndependenceCopula':
            output = 1/(2*torch.pi*self.sigma**2) * torch.exp(-(x**2+y**2)/(2*self.sigma**2))
            return output

    # log of density_x()
    def log_density(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        if self.target == 'GaussianCopula':
            x,y = x[...,0], x[...,1]
            output = -torch.log(torch.tensor(2*torch.pi)) - 0.5*torch.log(1-self.theta**2) - 2*torch.log(torch.tensor(self.sigma)) + \
                     + (2*self.theta*x*y - x**2 - y**2) / (2*self.sigma**2*(1-self.theta**2))
            return output
        else:
            u = normal2.cdf(x)
            x,y = x[...,0], x[...,1]
            output = self.log_density_u(u) - torch.log(torch.tensor(2*torch.pi)) - 2*torch.log(torch.tensor(self.sigma)) - (x**2+y**2) / (2*self.sigma**2)
            return output

    # gradient of log of density_x()
    def dx_logden(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        g = torch.zeros(x.shape).to(x)
        x,y = x[...,0], x[...,1]
        if self.target == 'GaussianCopula':
            g[...,0] = (self.theta*y-x) / (self.sigma**2*(1-self.theta**2))
            g[...,1] = (self.theta*x-y) / (self.sigma**2*(1-self.theta**2))
            return g
        elif self.target == 'ClaytonCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            g[...,0] = -(self.theta+1)/u + (1+2*self.theta)* v**self.theta / (u*v**self.theta+u**(self.theta+1)*(1-v**self.theta))
            g[...,1] = -(self.theta+1)/v + (1+2*self.theta)* u**self.theta / (v*u**self.theta+v**(self.theta+1)*(1-u**self.theta))
            g[...,0] = g[...,0] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-x**2/(2*self.sigma**2)) - x/self.sigma**2
            g[...,1] = g[...,1] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-y**2/(2*self.sigma**2)) - y/self.sigma**2
            return g
        elif self.target == 'GumbelCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            logu = torch.log(u)
            logv = torch.log(v)
            tmp = (-logu)**self.theta + (-logv)**self.theta
            tmpu = tmp**(1/self.theta-1) * (-logu)**(self.theta-1)
            tmpv = tmp**(1/self.theta-1) * (-logv)**(self.theta-1)
            g[...,0] = (-1 + (self.theta-1)/logu + tmpu + (2*self.theta-1)*(-logu)**(self.theta-1)/tmp - tmpu/(tmp**(1/self.theta)+self.theta-1)) / u
            g[...,1] = (-1 + (self.theta-1)/logv + tmpv + (2*self.theta-1)*(-logv)**(self.theta-1)/tmp - tmpv/(tmp**(1/self.theta)+self.theta-1)) / v
            g[...,0] = g[...,0] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-x**2/(2*self.sigma**2)) - x/self.sigma**2
            g[...,1] = g[...,1] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-y**2/(2*self.sigma**2)) - y/self.sigma**2
            return g
        elif self.target == 'FrankCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            tmp1 = torch.exp(-self.theta*u)
            tmp2 = torch.exp(-self.theta*v)
            tmp3 = torch.exp(-self.theta*(u+v))
            tmp4 = tmp1+tmp2-tmp3-torch.exp(-self.theta)
            g[...,0] = self.theta*(tmp1-tmp2-tmp3+torch.exp(-self.theta)) / tmp4
            g[...,1] = self.theta*(tmp2-tmp1-tmp3+torch.exp(-self.theta)) / tmp4
            g[...,0] = g[...,0] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-x**2/(2*self.sigma**2)) - x/self.sigma**2
            g[...,1] = g[...,1] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-y**2/(2*self.sigma**2)) - y/self.sigma**2
            return g
        elif self.target == 'JoeCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            ubar, vbar = 1-u, 1-v
            tempg = ubar**self.theta + vbar**self.theta - ubar**self.theta*vbar**self.theta
            dtempg_du = -self.theta*ubar**(self.theta-1)*(1-vbar**self.theta)
            dtempg_dv = -self.theta*vbar**(self.theta-1)*(1-ubar**self.theta)
            g[...,0] = -(self.theta-1)/ubar + dtempg_du*((1/self.theta-2)/tempg + 1/(self.theta-1+tempg))
            g[...,1] = -(self.theta-1)/vbar + dtempg_dv*((1/self.theta-2)/tempg + 1/(self.theta-1+tempg))
            g[...,0] = g[...,0] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-x**2/(2*self.sigma**2)) - x/self.sigma**2
            g[...,1] = g[...,1] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-y**2/(2*self.sigma**2)) - y/self.sigma**2
            return g
        elif self.target == 'StudentTCopula':
            rho, nu = self.theta[0], self.theta[1]
            u,v = normal2.cdf(x), normal2.cdf(y)
            s, w = _student_t_icdf(u, nu), _student_t_icdf(v, nu)
            A = 1 + (s**2 + w**2 - 2*rho*s*w) / (nu*(1-rho**2))
            B1, B2 = 1 + s**2/nu, 1 + w**2/nu
            # d/ds[log c(u,v)] and d/dw[log c(u,v)] -- hand-differentiated from the
            # log_density_u formula below; converted to d/du, d/dv via the reciprocal-density
            # identity d/du[t_nu^-1(u)] = 1/f_t(s;nu), same trick as _StudentTICDF.backward
            dlogc_ds = -(nu+2)*(s-rho*w)/(nu*(1-rho**2)*A) + (nu+1)*s/(nu*B1)
            dlogc_dw = -(nu+2)*(w-rho*s)/(nu*(1-rho**2)*A) + (nu+1)*w/(nu*B2)
            ft_s = torch.exp(torch.distributions.StudentT(nu).log_prob(s))
            ft_w = torch.exp(torch.distributions.StudentT(nu).log_prob(w))
            g[...,0] = dlogc_ds / ft_s
            g[...,1] = dlogc_dw / ft_w
            g[...,0] = g[...,0] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-x**2/(2*self.sigma**2)) - x/self.sigma**2
            g[...,1] = g[...,1] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-y**2/(2*self.sigma**2)) - y/self.sigma**2
            return g
        elif self.target == 'PlackettCopula':
            u,v = normal2.cdf(x), normal2.cdf(y)
            th, eta, S, D, N = self._plackett_parts(u, v)
            # log c = log(theta) + log(N) - 1.5 log(D), and dD/du = 2 eta (S - 2 theta v)
            g[...,0] = eta*(1-2*v)/N - 3*eta*(S - 2*th*v)/D
            g[...,1] = eta*(1-2*u)/N - 3*eta*(S - 2*th*u)/D
            g[...,0] = g[...,0] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-x**2/(2*self.sigma**2)) - x/self.sigma**2
            g[...,1] = g[...,1] / (torch.sqrt(torch.tensor(2*torch.pi))*self.sigma) * torch.exp(-y**2/(2*self.sigma**2)) - y/self.sigma**2
            return g
        elif self.target == 'IndependenceCopula':
            g[...,0] = -x/self.sigma**2
            g[...,1] = -y/self.sigma**2
            return g

    # ---- joint CDF and its two partials (h-functions) ---------------------
    # Needed by KhoudrajiCopula, whose CDF is a product of its components'
    # CDFs and whose density is a product-rule expansion in these four
    # quantities (C, dC/du, dC/dv, c). Every family here is EXCHANGEABLE,
    # C(u,v) = C(v,u), so hfunc_v is hfunc_u with the arguments swapped.
    # (Only the composite wrappers below, which can break exchangeability,
    # have to implement the two separately.)

    # theta is stored as float32, and a 0-dim float32 tensor does NOT promote
    # to float64 on its own: expm1(-theta) stays float32 even when u is
    # float64. Frank's h-function divides by expm1(-theta) + ea*eb, which
    # cancels down to ~exp(-theta) near the (1,1) corner, so float32 rounds
    # the e^-25 term away entirely and Kendall's tau of Frank(25) came out
    # 4.4e-3 too high. Casting to the input's dtype is a no-op for the float32
    # training path.
    def _theta_like(self, a):
        return None if self.theta is None else self.theta.to(a.dtype)

    # joint CDF C(u1, u2) on [0,1]^2
    def cdf_u(self, u):
        a, b = u[..., 0].clamp(_UEPS, 1 - _UEPS), u[..., 1].clamp(_UEPS, 1 - _UEPS)
        th = self._theta_like(a)
        if self.target == 'IndependenceCopula':
            return a * b
        elif self.target == 'GaussianCopula':
            return _bivariate_normal_cdf_t(normal.icdf(a), normal.icdf(b), th)
        elif self.target == 'StudentTCopula':
            rho, nu = th[0], th[1]
            return _bivariate_t_cdf_t(_student_t_icdf(a, nu), _student_t_icdf(b, nu), rho, nu)
        elif self.target == 'ClaytonCopula':
            inner = (a ** (-th) + b ** (-th) - 1).clamp_min(_UEPS)
            return inner ** (-1 / th)
        elif self.target == 'GumbelCopula':
            A = ((-torch.log(a)) ** th + (-torch.log(b)) ** th) ** (1 / th)
            return torch.exp(-A)
        elif self.target == 'FrankCopula':
            # log1p/expm1 form: the naive (exp(-t u)-1) cancels badly for small theta*u
            num = torch.expm1(-th * a) * torch.expm1(-th * b)
            return -1 / th * torch.log1p(num / torch.expm1(-th))
        elif self.target == 'JoeCopula':
            ab, bb = 1 - a, 1 - b
            S = ab ** th + bb ** th - ab ** th * bb ** th
            return 1 - S ** (1 / th)
        elif self.target == 'PlackettCopula':
            # C is the smaller root of the quadratic with discriminant D. The
            # eta = 0 case is the one genuine 0/0 in the family (the density
            # and h-function are both fine there), so it gets a Python-level
            # branch on the fixed parameter rather than a nudge.
            if abs(float(th) - 1.0) < 1e-12:
                return a * b
            _, eta, S, D, _ = self._plackett_parts(a, b)
            return (S - torch.sqrt(D)) / (2 * eta)

    # dC/du1 -- the conditional CDF P(U2 <= u2 | U1 = u1)
    def hfunc_u(self, u):
        a, b = u[..., 0].clamp(_UEPS, 1 - _UEPS), u[..., 1].clamp(_UEPS, 1 - _UEPS)
        th = self._theta_like(a)
        if self.target == 'IndependenceCopula':
            return b
        elif self.target == 'GaussianCopula':
            za, zb = normal.icdf(a), normal.icdf(b)
            return torch.special.ndtr((zb - th * za) / torch.sqrt(1 - th ** 2))
        elif self.target == 'StudentTCopula':
            rho, nu = th[0], th[1]
            sa, sb = _student_t_icdf(a, nu), _student_t_icdf(b, nu)
            arg = (sb - rho * sa) / torch.sqrt((1 - rho ** 2) * (nu + sa ** 2) / (nu + 1))
            return _student_t_cdf(arg, nu + 1)
        elif self.target == 'ClaytonCopula':
            inner = (a ** (-th) + b ** (-th) - 1).clamp_min(_UEPS)
            return a ** (-th - 1) * inner ** (-1 / th - 1)
        elif self.target == 'GumbelCopula':
            la, lb = -torch.log(a), -torch.log(b)
            S = la ** th + lb ** th
            A = S ** (1 / th)
            return torch.exp(-A) * A ** (1 - th) * la ** (th - 1) / a
        elif self.target == 'FrankCopula':
            ea, eb = torch.expm1(-th * a), torch.expm1(-th * b)
            return torch.exp(-th * a) * eb / (torch.expm1(-th) + ea * eb)
        elif self.target == 'JoeCopula':
            ab, bb = 1 - a, 1 - b
            S = ab ** th + bb ** th - ab ** th * bb ** th
            return ab ** (th - 1) * (1 - bb ** th) * S ** (1 / th - 1)
        elif self.target == 'PlackettCopula':
            # dC/du = (1/2)[1 - (S - 2 theta v)/sqrt(D)] -- finite at eta = 0,
            # where it collapses to v, so no special case needed here
            th_, eta, S, D, _ = self._plackett_parts(a, b)
            return 0.5 * (1 - (S - 2 * th_ * b) / torch.sqrt(D))

    # dC/du2; every family in this class is exchangeable, so it is hfunc_u
    # with the coordinates swapped
    def hfunc_v(self, u):
        return self.hfunc_u(u.flip(-1))

    # used on log_density() to verify that dx_logden is right
    def test(self,x):
        return jacrev(self.log_density)(x)

    # copula density [0,1]^2 -> R^+
    def density_u(self, u):
        if self.target == 'GaussianCopula':
            tmp = normal.icdf(u)
            x, y = tmp[..., 0], tmp[..., 1]
            output = 1/(torch.sqrt(1-self.theta**2))*torch.exp((2*self.theta*x*y-self.theta**2*(x**2+y**2))/(2*(1-self.theta**2)))
            return output
        elif self.target == 'ClaytonCopula':
            u, v = u[...,0], u[...,1]
            output = (1+self.theta)*(u*v)**(-self.theta-1)*(u**(-self.theta)+v**(-self.theta)-1)**(-1/self.theta-2)
            return output
        elif self.target == 'GumbelCopula':
            u,v = u[...,0], u[...,1]
            logu = -torch.log(u)
            logv = -torch.log(v)
            tmp = (logu)**self.theta + (logv)**self.theta
            output = 1/(u*v)*(logu * logv)**(self.theta-1)*torch.exp(-tmp**(1/self.theta))*tmp**(1/self.theta-2)*(tmp**(1/self.theta)+self.theta-1)
            return output
        elif self.target == 'FrankCopula':
            u,v = u[...,0], u[...,1]
            output = self.theta*(1-torch.exp(-self.theta))*torch.exp(-self.theta*(u+v)) / (torch.exp(-self.theta*u)+torch.exp(-self.theta*v)-
                                                                            torch.exp(-self.theta*(u+v))-torch.exp(-self.theta))**2
            return output
        elif self.target == 'JoeCopula':
            u,v = u[...,0], u[...,1]
            tempg = (1-u)**self.theta + (1-v)**self.theta - (1-u)**self.theta*(1-v)**self.theta
            output = (1-u)**(self.theta-1) * (1-v)**(self.theta-1) * tempg**(1/self.theta-2)*(self.theta-1+tempg)
            return output
        elif self.target == 'StudentTCopula':
            rho, nu = self.theta[0], self.theta[1]
            u, v = u[...,0], u[...,1]
            s, w = _student_t_icdf(u, nu), _student_t_icdf(v, nu)
            A = 1 + (s**2 + w**2 - 2*rho*s*w) / (nu*(1-rho**2))
            B1, B2 = 1 + s**2/nu, 1 + w**2/nu
            K = torch.exp(torch.lgamma((nu+2)/2) + torch.lgamma(nu/2) - 2*torch.lgamma((nu+1)/2))
            output = K / torch.sqrt(1-rho**2) * A**(-(nu+2)/2) * B1**((nu+1)/2) * B2**((nu+1)/2)
            return output
        elif self.target == 'PlackettCopula':
            u, v = u[...,0], u[...,1]
            th, eta, S, D, N = self._plackett_parts(u, v)
            return th * N / D**1.5
        elif self.target == 'IndependenceCopula':
            return torch.ones_like(u[...,0])

    # log of copula density [0,1]^2 -> R
    def log_density_u(self, u):
        if self.target == 'GaussianCopula':
            tmp = normal.icdf(u)
            x, y = tmp[..., 0], tmp[..., 1]
            output = -0.5 * torch.log(1 - self.theta ** 2)+(2*self.theta * x * y - self.theta ** 2 * (x ** 2 + y ** 2)) / (2 * (1 - self.theta ** 2))
            return output
        elif self.target == 'ClaytonCopula':
            u, v = u[...,0], u[...,1]
            output = torch.log(1+self.theta)-(self.theta+1)*(torch.log(u)+torch.log(v))-(1/self.theta+2)*torch.log(u**(-self.theta)+v**(-self.theta)-1)
            return output
        elif self.target == 'GumbelCopula':
            u,v = u[...,0], u[...,1]
            logu = torch.log(u)
            logv = torch.log(v)
            tmp = (-logu)**self.theta + (-logv)**self.theta
            output =-logu-logv+(self.theta-1)*(torch.log(-logu)+torch.log(-logv))-tmp**(1/self.theta)+(1/self.theta-2)*torch.log(tmp)+\
                     torch.log(tmp**(1/self.theta)+self.theta-1)
            return output
        elif self.target == 'FrankCopula':
            u,v = u[...,0], u[...,1]
            output = torch.log(self.theta*(1-torch.exp(-self.theta)))-self.theta*(u+v)-\
                     torch.log((torch.exp(-self.theta*u)+torch.exp(-self.theta*v)-torch.exp(-self.theta*(u+v))-torch.exp(-self.theta))**2)
            return output
        elif self.target == 'JoeCopula':
            u,v = u[...,0], u[...,1]
            tempg = (1-u)**self.theta + (1-v)**self.theta - (1-u)**self.theta*(1-v)**self.theta
            output = (self.theta-1)*(torch.log(1-u)+torch.log(1-v)) + (1/self.theta-2)*torch.log(tempg) + torch.log(self.theta-1+tempg)
            return output
        elif self.target == 'StudentTCopula':
            rho, nu = self.theta[0], self.theta[1]
            u, v = u[...,0], u[...,1]
            s, w = _student_t_icdf(u, nu), _student_t_icdf(v, nu)
            A = 1 + (s**2 + w**2 - 2*rho*s*w) / (nu*(1-rho**2))
            B1, B2 = 1 + s**2/nu, 1 + w**2/nu
            output = torch.lgamma((nu+2)/2) + torch.lgamma(nu/2) - 2*torch.lgamma((nu+1)/2) - 0.5*torch.log(1-rho**2) \
                     - (nu+2)/2*torch.log(A) + (nu+1)/2*torch.log(B1) + (nu+1)/2*torch.log(B2)
            return output
        elif self.target == 'PlackettCopula':
            u, v = u[...,0], u[...,1]
            th, eta, S, D, N = self._plackett_parts(u, v)
            return torch.log(th) + torch.log(N) - 1.5*torch.log(D)
        elif self.target == 'IndependenceCopula':
            return torch.zeros_like(u[...,0])

    # draw n samples from copula
    def sample_u(self, n_samples):
        if self.target == 'GaussianCopula':
            sample = GaussianCopula(self.theta.numpy()).rvs(n_samples)
        elif self.target == 'ClaytonCopula':
            sample = ClaytonCopula(self.theta.numpy()).rvs(n_samples)
        elif self.target == 'GumbelCopula':
            sample = GumbelCopula(self.theta.numpy()).rvs(n_samples)
        elif self.target == 'FrankCopula':
            # random_state MUST be a Generator: statsmodels draws the frailty
            # from logser(1 - e^-theta), and the legacy RandomState.logseries
            # (what the global np.random gives) is wrong that close to p=1 --
            # capped at 2^31 on Windows and biased low throughout. Measured at
            # theta=25: P(both > 0.98) = 0 vs 0.007 in the (1,1) corner, which
            # a radially symmetric copula must match against (0,0). Seeded from
            # the global RNG so np.random.seed still makes it reproducible.
            # theta MUST be float64 here: in float32, 1 - e^-theta rounds to
            # exactly 1.0 for theta >~ 17, and scipy's logser rejects p = 1
            # (Frank(25) raised a domain error).
            rng = np.random.default_rng(np.random.randint(2**32, dtype=np.uint64))
            sample = FrankCopula(np.float64(self.theta.item())).rvs(n_samples, random_state=rng)
        elif self.target == 'JoeCopula':
            # Marshall-Olkin: mix n_samples pairs of independent Uniform(0,1)
            # through the Joe generator inverse phi^-1(s;theta) = 1-(1-e^-s)^(1/theta),
            # using a shared-per-sample Sibuya(1/theta) frailty variable v --
            # same construction as ClaytonCopula/GumbelCopula.rvs() in statsmodels,
            # just with Joe's generator and frailty distribution instead
            theta = self.theta.item()
            x = np.random.random((n_samples, 2))
            v = _rvs_sibuya(1 / theta, n_samples)[:, None]
            s = -np.log(x) / v
            # -expm1(-s), NOT 1-exp(-s): Sibuya(1/theta) has a very heavy tail
            # (infinite mean for theta>1), so v routinely reaches ~1e16+ and s
            # underflows to ~1e-20. There 1-exp(-s) cancels to exactly 0, and
            # the sample saturates at u=1 exactly -- a spurious point mass ON
            # the boundary, outside the open unit square where the copula
            # density lives (log_density_u would return -inf there). Measured
            # rate of exact-1.0 entries with the naive form: 0% at theta=2,
            # 0.008% at theta=4, 0.90% at theta=8, which is what broke the
            # marginal-uniformity KS test at theta=8.
            sample = 1 - (-np.expm1(-s)) ** (1 / theta)
        elif self.target == 'StudentTCopula':
            # StudentTCopula is an EllipticalCopula, whose .rvs() (unlike
            # ArchimedeanCopula's) IS generic -- no hand-rolled sampler needed here
            rho, nu = self.theta[0].item(), self.theta[1].item()
            sample = StudentTCopula(corr=rho, df=nu).rvs(n_samples)
        elif self.target == 'PlackettCopula':
            # Conditional inversion in closed form (Johnson 1987): the
            # conditional CDF h(v|u) = t is a quadratic in v, so no numerical
            # root-finding and no rejection -- unlike Joe's frailty sampler.
            # Reduces to v = t at theta = 1, as it must.
            th = self.theta.item()
            u, t = np.random.random(n_samples), np.random.random(n_samples)
            a = t * (1 - t)
            b = th + a * (th - 1) ** 2
            cc = 2 * a * (u * th ** 2 + 1 - u) + th * (1 - 2 * a)
            d = np.sqrt(th) * np.sqrt(th + 4 * a * u * (1 - u) * (1 - th) ** 2)
            v = (cc - (1 - 2 * t) * d) / (2 * b)
            sample = np.stack([u, np.clip(v, 0.0, 1.0)], axis=1)
        elif self.target == 'IndependenceCopula':
            sample = IndependenceCopula().rvs(n_samples)
        return sample

    # draw n samples from f
    def sample_x(self, n_samples):
        sample = self.sample_u(n_samples)
        sample = norm.ppf(sample,loc=0, scale=self.sigma)
        return sample


class MixtureCopula(nn.Module):
    """Copula formed as a convex combination of other copulas:
    c(u) = sum_i w_i * c_i(u), with w_i >= 0 and sum_i w_i = 1.

    A convex combination of valid copulas is itself a valid copula, so any 
    mix of the families supported by `load_target`, including nested 
    `MixtureCopula`s, can be used here.
    All components must share the same marginal `sigma`, per the same
    Sklar-type decomposition constraint as `load_target`.

    Implements the same interface as `load_target` (sigma, density_x,
    log_density, dx_logden, density_u, log_density_u, sample_u, sample_x),
    so it's a drop-in replacement anywhere a `load_target` instance is used.

    components: list of `load_target` instances, or (target, theta) tuples
        to be built via `load_target(target, theta, Sigma)`.
    weights: nonnegative weights, one per component; normalized to sum to 1.
    Sigma: shared marginal std, required if any component is given as a
        (target, theta) tuple rather than an already-built `load_target`.

    This class was implemented with Claude Code.
    """

    def __init__(self, components, weights, Sigma=None):
        super().__init__()
        is_built = lambda c: isinstance(c, nn.Module)  # load_target/MixtureCopula, as opposed to a (target, theta) tuple
        if any(not is_built(c) for c in components) and Sigma is None:
            raise ValueError("Sigma must be given when components include (target, theta) tuples")
        built = [c if is_built(c) else load_target(c[0], c[1], Sigma) for c in components]

        sigmas = {float(c.sigma) for c in built}
        if len(sigmas) != 1:
            raise ValueError("all components of a MixtureCopula must share the same sigma")
        self.components = nn.ModuleList(built)
        self.sigma = built[0].sigma
        self.target = "MixtureCopula"
        self.theta = None

        w = torch.as_tensor(weights, dtype=torch.get_default_dtype())
        if w.numel() != len(self.components):
            raise ValueError("need exactly one weight per component")
        if torch.any(w < 0):
            raise ValueError("weights must be nonnegative")
        self.weights = w / w.sum()  # normalized so the combination stays a valid copula

    # per-component quantity stacked along a new leading (component) dim
    def _stack(self, fn_name, arg):
        return torch.stack([getattr(c, fn_name)(arg) for c in self.components], dim=0)

    # reshapes/casts a (K,) tensor to broadcast against a (K, *batch) tensor
    def _broadcast(self, weights, values):
        return weights.to(values).view((-1,) + (1,) * (values.dim() - 1))

    # copula density [0,1]^2 -> R^+: exact linear combination
    def density_u(self, u):
        vals = self._stack("density_u", u)
        return (self._broadcast(self.weights, vals) * vals).sum(dim=0)

    # log of density_u(): NOT a linear combination of the component
    # log-densities (log of a sum != sum of logs) -- via logsumexp instead
    def log_density_u(self, u):
        logs = self._stack("log_density_u", u)
        return torch.logsumexp(logs + self._broadcast(torch.log(self.weights), logs), dim=0)

    # joint density f(x1, x2): still a linear combination since the
    # marginal (sigma) is shared across all components
    def density_x(self, x):
        vals = self._stack("density_x", x)
        return (self._broadcast(self.weights, vals) * vals).sum(dim=0)

    def log_density(self, x):
        logs = self._stack("log_density", x)
        return torch.logsumexp(logs + self._broadcast(torch.log(self.weights), logs), dim=0)

    # gradient of log_density(): d/dx log(sum_i w_i f_i) = sum_i r_i * dx_logden_i,
    # a responsibility-weighted average of the component gradients, where
    # r_i = w_i f_i(x) / sum_j w_j f_j(x) is the posterior mixture weight at x
    def dx_logden(self, x):
        logs = self._stack("log_density", x)  # (K, *batch)
        log_w = self._broadcast(torch.log(self.weights), logs)
        log_mix = torch.logsumexp(logs + log_w, dim=0, keepdim=True)  # (1, *batch)
        resp = torch.exp(logs + log_w - log_mix)  # (K, *batch)
        grads = torch.stack([c.dx_logden(x) for c in self.components], dim=0)  # (K, *batch, 2)
        return (resp.unsqueeze(-1) * grads).sum(dim=0)

    # CDF and h-functions are linear in the components, exactly like density_u
    # (differentiation w.r.t. u commutes with the convex combination)
    def cdf_u(self, u):
        vals = self._stack("cdf_u", u)
        return (self._broadcast(self.weights, vals) * vals).sum(dim=0)

    def hfunc_u(self, u):
        vals = self._stack("hfunc_u", u)
        return (self._broadcast(self.weights, vals) * vals).sum(dim=0)

    def hfunc_v(self, u):
        vals = self._stack("hfunc_v", u)
        return (self._broadcast(self.weights, vals) * vals).sum(dim=0)

    # used on log_density() to verify that dx_logden is right
    def test(self, x):
        return jacrev(self.log_density)(x)

    # draw n samples from the mixture: assign each sample to a component
    # according to `weights`, then sample from that component
    def sample_u(self, n_samples):
        counts = np.random.multinomial(n_samples, self.weights.detach().cpu().numpy())
        parts = [c.sample_u(int(n)) for c, n in zip(self.components, counts) if n > 0]
        sample = np.concatenate(parts, axis=0)
        np.random.shuffle(sample)
        return sample

    # draw n samples from f
    def sample_x(self, n_samples):
        sample = self.sample_u(n_samples)
        sample = norm.ppf(sample, loc=0, scale=self.sigma)
        return sample


class MirroredCopula(nn.Module):
    """
    Copula mirrored about the midline of one or both coordinates.

    For mirrored axes `axes`, the density is

        c_mir(u, v) = c(u', v'),   u' = 1-u if 0 in axes else u,
                                   v' = 1-v if 1 in axes else v

    which is still a valid copula (the reflection is a measure-preserving
    bijection of [0,1]^2 mapping uniform marginals to uniform marginals), so
    a mirrored family can be used anywhere `load_target` can, including as a
    `MixtureCopula` component.

    The copula literature calls these the 90/180/270-degree ROTATIONS, and
    that name is a misnomer for what this construction does: mirroring axis 0
    is its 90, mirroring both is its 180 (the survival copula), mirroring
    axis 1 is its 270. A true rotation is a reflection composed with a
    TRANSPOSE, c(v, 1-u), and this class is only the reflection half; the
    other half is `TransposedCopula`, and its docstring gives the three
    compositions that build the genuine rotations. The two coincide for
    exchangeable copulas, which is every `load_target` family -- but
    `KhoudrajiCopula` and `GluedCopula` are not exchangeable, and there the
    difference is plainly visible (mirroring a glued chevron cannot turn it,
    only reflect it; see the gallery notebook).

    Only worth doing for families that are not already reflection-symmetric:
    Gaussian, Student-t, Frank and Independence all satisfy
    c(1-u, 1-v) = c(u, v), so mirroring *both* their coordinates is a no-op
    (mirroring a single one is *not*, and flips a positive dependence into a
    negative one). The asymmetric Archimedeans -- Clayton (lower-tail
    dependence), Gumbel and Joe (upper-tail) -- are the interesting case:
    mirroring both coordinates swaps which tail the dependence lives in.

    In the lifted R^2 space the reflection is just a sign flip of the
    corresponding coordinate, since the shared marginal Normal(0, sigma) is
    symmetric: 1 - Phi_sigma(x) = Phi_sigma(-x). So

        f_mir(x, y) = f(s_x * x, s_y * y),  s = -1 on mirrored coordinates,

    with unit Jacobian -- hence density_x/log_density compose by negating
    the input, and dx_logden picks up one factor of s per coordinate by the
    chain rule. No family-specific formula is needed here, which is why this
    is a wrapper rather than new branches in `load_target`.

    base: any object with the `load_target` interface (`load_target`,
        `MixtureCopula`, or another `MirroredCopula`).
    axes: index of the coordinate to mirror (0 or 1), a tuple/list of both,
        or () for the identity. Coordinate INDICES, not flags -- `axes=1`
        mirrors the second coordinate, it does not mean "mirror the first".
    """

    def __init__(self, base, axes=(0, 1)):
        super().__init__()
        if axes is None:
            axes = ()
        if isinstance(axes, (bool, np.bool_, int, np.integer)):
            axes = (axes,)
        axes = tuple(axes)
        if any(isinstance(a, (bool, np.bool_)) for a in axes):
            raise ValueError("axes are coordinate indices, not flags: pass "
                             "axes=0, 1, (0, 1) or ()")
        if any(int(a) not in (0, 1) for a in axes):
            raise ValueError("axes must be 0 and/or 1 (this is the bivariate construction)")
        axes = tuple(int(a) for a in axes)
        flip = tuple(i in axes for i in range(2))

        # collapse nested mirrors, so composing them never deepens the call chain
        if isinstance(base, MirroredCopula):
            flip = tuple(f ^ g for f, g in zip(flip, base.flip))
            base = base.base

        self.base = base
        # per-coordinate bool form of `axes`, which is what the maps below want
        self.flip = flip
        self.sigma = base.sigma
        self.target = "MirroredCopula"
        self.theta = base.theta
        # x-space image of the u-space reflection (see class docstring)
        self.register_buffer("signs", torch.tensor([-1.0 if f else 1.0 for f in flip]))

    @property
    def axes(self):
        """The mirrored coordinate indices: (), (0,), (1,) or (0, 1)."""
        return tuple(i for i, f in enumerate(self.flip) if f)

    @property
    def axes_tag(self):
        """Short label for plot titles and reports: 'u0', 'u1', 'u0u1', 'none'."""
        return "".join(f"u{i}" for i in self.axes) or "none"

    def extra_repr(self):
        return f"axes={self.axes}"

    # (u, v) -> (u', v'): reflect the flagged coordinates about 1/2
    def _flip_u(self, u):
        if not any(self.flip):
            return u
        return torch.stack([1 - u[..., i] if f else u[..., i]
                            for i, f in enumerate(self.flip)], dim=-1)

    # (x, y) -> (s_x x, s_y y): the same reflection in the lifted space
    def _flip_x(self, x):
        if not any(self.flip):
            return x
        return x * self.signs.to(x)

    def density_u(self, u):
        return self.base.density_u(self._flip_u(u))

    def log_density_u(self, u):
        return self.base.log_density_u(self._flip_u(u))

    def density_x(self, x):
        return self.base.density_x(self._flip_x(x))

    def log_density(self, x):
        return self.base.log_density(self._flip_x(x))

    # chain rule through x -> s*x: grad_x log f(s x) = s * (dx_logden)(s x)
    def dx_logden(self, x):
        return self.signs.to(x) * self.base.dx_logden(self._flip_x(x))

    # C_rot from the base C: flipping u contributes the C(1,v)=v term,
    # flipping v the C(u,1)=u term, and flipping both subtracts the doubly
    # counted 1 (the same algebra the fade script's cdf branch uses).
    def cdf_u(self, u):
        fu, fv = self.flip
        base = self.base.cdf_u(self._flip_u(u))
        out = (-1.0) ** (fu + fv) * base
        if fu:
            out = out + u[..., 1]
        if fv:
            out = out + u[..., 0]
        if fu and fv:
            out = out - 1.0
        return out

    # dC_rot/du = P(V' <= v | U' = u): reflecting V turns that into its
    # complement, so hfunc_u picks up the 1 - (.) exactly when `v` is flipped
    # (and vice versa for hfunc_v) -- note the crossed dependence.
    def hfunc_u(self, u):
        h = self.base.hfunc_u(self._flip_u(u))
        return 1.0 - h if self.flip[1] else h

    def hfunc_v(self, u):
        h = self.base.hfunc_v(self._flip_u(u))
        return 1.0 - h if self.flip[0] else h

    # used on log_density() to verify that dx_logden is right
    def test(self, x):
        return jacrev(self.log_density)(x)

    # draw n samples from copula: reflect the base family's draws
    def sample_u(self, n_samples):
        sample = np.asarray(self.base.sample_u(n_samples), dtype=float).copy()
        for i, f in enumerate(self.flip):
            if f:
                sample[..., i] = 1.0 - sample[..., i]
        return sample

    # draw n samples from f
    def sample_x(self, n_samples):
        sample = self.sample_u(n_samples)
        sample = norm.ppf(sample, loc=0, scale=self.sigma)
        return sample


class TransposedCopula(nn.Module):
    """Copula with its two coordinates exchanged: c_T(u, v) = c(v, u).

    The transpose of a copula is a copula (swapping coordinates is a
    measure-preserving bijection of [0,1]^2 that maps the pair of uniform
    marginals to itself, just exchanged), so a transposed family can be used
    anywhere `load_target` can, including as a component of any other wrapper
    here.

    It is a NO-OP for every exchangeable copula, which is every `load_target`
    family and any `MixtureCopula` of them. The two constructions in this file
    that break exchangeability -- `KhoudrajiCopula` (smoothly, via unequal
    exponents) and `GluedCopula` (non-smoothly, via a seam along one
    coordinate) -- are the only ones for which this class does anything, and
    they are the reason it exists.

    WHY IT IS HERE: to complete `MirroredCopula` into the true rotations.
    What the copula literature calls the 90/270-degree rotations is a
    reflection composed with a transpose, and `MirroredCopula` supplies only
    the reflection half (see its docstring). With this class the genuine
    rotations of the unit square are available as compositions -- writing
    M_i for `MirroredCopula(., i)` and T for this class:

        90 deg CCW,  (u,v) -> (1-v, u):   c_rot(u,v) = c(v, 1-u) = M_0(T(c))
        180 deg,     (u,v) -> (1-u, 1-v): c_rot(u,v) = c(1-u,1-v) = M_(0,1)(c)
        90 deg CW,   (u,v) -> (v, 1-u):   c_rot(u,v) = c(1-v, u) = M_1(T(c))

    (the 180 row needs no transpose, which is why the reflection-only wrapper
    is exactly right there -- it is the survival copula either way). Mirror
    and transpose do NOT commute; they satisfy

        T . M_axes = M_(swapped axes) . T

    so `MirroredCopula(TransposedCopula(c), 0)` and
    `TransposedCopula(MirroredCopula(c, 1))` are the same copula, while
    `TransposedCopula(MirroredCopula(c, 0))` is the OTHER rotation. Nothing
    canonicalises the nesting order, so build a rotation from one of the two
    forms written above rather than assuming an ordering is normalised away.

    In the lifted R^2 space the transpose is again just a coordinate swap --
    both coordinates share the same marginal Normal(0, sigma), so
    f_T(x, y) = f(y, x) with unit Jacobian -- hence every method composes with
    a swap of the last axis and needs no family-specific formula. The two
    h-functions swap both their arguments and their roles: dC_T/du at (u,v) is
    dC/dv at (v,u).

    base: any object with the `load_target` interface (`load_target`,
        `MixtureCopula`, `MirroredCopula`, `KhoudrajiCopula`, `GluedCopula`,
        or another `TransposedCopula`). Transposing twice collapses back to
        an identity wrapper on the inner base, the way nested mirrors do.

    This class was implemented with Claude Code.
    """

    def __init__(self, base):
        super().__init__()
        swapped = True
        # collapse the involution, so composing transposes never deepens the
        # call chain (T . T = id); the result is an identity wrapper, exactly
        # as MirroredCopula(base, ()) is
        if isinstance(base, TransposedCopula):
            swapped = not base.swapped
            base = base.base

        self.base = base
        self.swapped = swapped
        self.sigma = base.sigma
        self.target = "TransposedCopula"
        self.theta = base.theta

    def extra_repr(self):
        return f"swapped={self.swapped}"

    # (a, b) -> (b, a) on the last axis, in either space -- the same swap
    # works on u and on x (see the class docstring)
    def _swap(self, z):
        if not self.swapped:
            return z
        return torch.flip(z, dims=(-1,))

    # pick the base method that plays the role this one does after the swap:
    # dC_T/du = dC/dv and vice versa
    def _hname(self, name):
        if not self.swapped:
            return name
        return "hfunc_v" if name == "hfunc_u" else "hfunc_u"

    def density_u(self, u):
        return self.base.density_u(self._swap(u))

    def log_density_u(self, u):
        return self.base.log_density_u(self._swap(u))

    def density_x(self, x):
        return self.base.density_x(self._swap(x))

    def log_density(self, x):
        return self.base.log_density(self._swap(x))

    # chain rule through the swap S, which is its own inverse and symmetric:
    # grad_x log f(S x) = S (dx_logden)(S x)
    def dx_logden(self, x):
        return self._swap(self.base.dx_logden(self._swap(x)))

    # C_T(u, v) = C(v, u) directly -- no extra terms, unlike the mirrored CDF
    def cdf_u(self, u):
        return self.base.cdf_u(self._swap(u))

    def hfunc_u(self, u):
        return getattr(self.base, self._hname("hfunc_u"))(self._swap(u))

    def hfunc_v(self, u):
        return getattr(self.base, self._hname("hfunc_v"))(self._swap(u))

    # used on log_density() to verify that dx_logden is right
    def test(self, x):
        return jacrev(self.log_density)(x)

    # draw n samples from copula: swap the base family's draws
    def sample_u(self, n_samples):
        sample = np.asarray(self.base.sample_u(n_samples), dtype=float)
        if not self.swapped:
            return sample.copy()
        return sample[..., ::-1].copy()

    # draw n samples from f
    def sample_x(self, n_samples):
        sample = self.sample_u(n_samples)
        return norm.ppf(sample, loc=0, scale=self.sigma)


class KhoudrajiCopula(nn.Module):
    """Khoudraji's device: an ASYMMETRIC copula built from two symmetric ones.

        K(u, v) = C1(u^(1-a1), v^(1-a2)) * C2(u^a1, v^a2),    a1, a2 in [0, 1]

    Every family in `load_target` is exchangeable (C(u,v) = C(v,u)) and all
    but the asymmetric Archimedeans are radially symmetric too, so none of them
    can express "the upper tail is coupled more tightly than the lower one"
    in a *direction-dependent* way. Khoudraji's device is the standard fix:
    raising the two coordinates to different powers breaks the exchangeability
    while preserving the copula property, so `K(u,v) != K(v,u)` whenever
    a1 != a2 and the two components differ.

    Why it stays a copula: the marginals come out uniform for free,
    K(u,1) = C1(u^(1-a1), 1) * C2(u^a1, 1) = u^(1-a1) * u^a1 = u, because the
    two exponent families multiply back to the identity. (This is Liebscher's
    general construction with g_1(u) = u^(1-a), g_2(u) = u^a.) In two
    dimensions the result is a genuine copula for any copulas C1, C2 and any
    a1, a2 in [0,1]; two shapes are pinned against pdf quadrature by
    `self_check` in fade_distance.py.

    Degenerate shapes collapse as you would expect: a1 = a2 = 0 gives C1
    alone, a1 = a2 = 1 gives C2 alone.

    Density (product rule on the two factors, all four terms needed):

        c_K = p_u q_v c1(p,q) C2(r,s)  +  p_u s_v h1_u(p,q) h2_v(r,s)
            + r_u q_v h1_v(p,q) h2_u(r,s)  +  r_u s_v C1(p,q) c2(r,s)

    with p = u^(1-a1), r = u^a1, q = v^(1-a2), s = v^a2 and their derivatives
    p_u = (1-a1) u^(-a1), r_u = a1 u^(a1-1) (likewise q_v, s_v). This is why
    `cdf_u`/`hfunc_u`/`hfunc_v` had to be added to the other classes.

    base1, base2: any objects with the `load_target` interface. `base2`
        defaults to an Independence copula with the same sigma, which is the
        usual textbook form K(u,v) = C(u^(1-a1), v^(1-a2)) * u^a1 * v^a2.
    shape: (a1, a2) in [0,1]^2, or a single float used for both.

    This class was implemented with Claude Code
    """

    def __init__(self, base1, base2=None, shape=(0.5, 0.5)):
        super().__init__()
        if np.isscalar(shape):
            shape = (float(shape), float(shape))
        a1, a2 = (float(shape[0]), float(shape[1]))
        if not (0.0 <= a1 <= 1.0 and 0.0 <= a2 <= 1.0):
            raise ValueError("Khoudraji shape parameters must lie in [0, 1]")
        if base2 is None:
            base2 = load_target('IndependenceCopula', None, base1.sigma)
        if float(base1.sigma) != float(base2.sigma):
            raise ValueError("both components of a KhoudrajiCopula must share the same sigma")

        self.base1, self.base2 = base1, base2
        self.shape = (a1, a2)
        self.sigma = base1.sigma
        self.target = "KhoudrajiCopula"
        self.theta = None

    def extra_repr(self):
        return f"shape=({self.shape[0]:g}, {self.shape[1]:g})"

    # the four powered arguments and their derivatives. The Python-level
    # branches on a1/a2 (fixed scalars, never tensors) keep the degenerate
    # shapes exact: at a1 = 0 the r-terms must vanish, but u^(a1-1) is
    # infinite at u = 0, and 0 * inf would be a NaN rather than a 0.
    def _parts(self, u):
        a1, a2 = self.shape
        x = u[..., 0].clamp(_UEPS, 1.0)
        y = u[..., 1].clamp(_UEPS, 1.0)
        zero = torch.zeros_like(x)
        p, r = x ** (1 - a1), x ** a1
        q, s = y ** (1 - a2), y ** a2
        p_u = (1 - a1) * x ** (-a1) if a1 < 1 else zero
        r_u = a1 * x ** (a1 - 1) if a1 > 0 else zero
        q_v = (1 - a2) * y ** (-a2) if a2 < 1 else zero
        s_v = a2 * y ** (a2 - 1) if a2 > 0 else zero
        return (torch.stack([p, q], -1), torch.stack([r, s], -1),
                p_u, q_v, r_u, s_v)

    def cdf_u(self, u):
        pq, rs, *_ = self._parts(u)
        return self.base1.cdf_u(pq) * self.base2.cdf_u(rs)

    # A term whose prefactor vanishes identically is SKIPPED, never merely
    # multiplied by zero

    # dK/du and dK/dv: product rule on the two factors (two terms each)
    def hfunc_u(self, u):
        a1, a2 = self.shape
        pq, rs, p_u, q_v, r_u, s_v = self._parts(u)
        out = torch.zeros_like(p_u)
        if a1 < 1:
            out = out + self.base1.hfunc_u(pq) * p_u * self.base2.cdf_u(rs)
        if a1 > 0:
            out = out + self.base1.cdf_u(pq) * self.base2.hfunc_u(rs) * r_u
        return out

    def hfunc_v(self, u):
        a1, a2 = self.shape
        pq, rs, p_u, q_v, r_u, s_v = self._parts(u)
        out = torch.zeros_like(q_v)
        if a2 < 1:
            out = out + self.base1.hfunc_v(pq) * q_v * self.base2.cdf_u(rs)
        if a2 > 0:
            out = out + self.base1.cdf_u(pq) * self.base2.hfunc_v(rs) * s_v
        return out

    # copula density: the four-term expansion in the class docstring
    def density_u(self, u):
        a1, a2 = self.shape
        pq, rs, p_u, q_v, r_u, s_v = self._parts(u)
        out = torch.zeros_like(p_u)
        if a1 < 1 and a2 < 1:
            out = out + p_u * q_v * self.base1.density_u(pq) * self.base2.cdf_u(rs)
        if a1 < 1 and a2 > 0:
            out = out + p_u * s_v * self.base1.hfunc_u(pq) * self.base2.hfunc_v(rs)
        if a1 > 0 and a2 < 1:
            out = out + r_u * q_v * self.base1.hfunc_v(pq) * self.base2.hfunc_u(rs)
        if a1 > 0 and a2 > 0:
            out = out + r_u * s_v * self.base1.cdf_u(pq) * self.base2.density_u(rs)
        return out

    # no cancellation-free closed form here: the density is a sum of four
    # nonnegative products, so the honest route is the log of the sum
    def log_density_u(self, u):
        return torch.log(self.density_u(u).clamp_min(1e-300))

    def density_x(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        return self.density_u(normal2.cdf(x)) * torch.exp(
            normal2.log_prob(x[..., 0]) + normal2.log_prob(x[..., 1]))

    def log_density(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        return (self.log_density_u(normal2.cdf(x))
                + normal2.log_prob(x[..., 0]) + normal2.log_prob(x[..., 1]))

    # Unlike every other class here, this one does NOT hand-derive the
    # gradient: log_density already routes through four product-rule terms per
    # component, and chain-ruling that by hand would be far more error-prone
    # than valuable. torch.autograd.grad on the explicit formula is exact (not
    # a finite difference) and the base pieces are all differentiable -- the
    # bivariate normal/t CDFs supply analytic backwards via
    # _BivariateNormalCDF/_BivariateTCDF. The graph is local to this call, so
    # nothing leaks into the training loss.
    def dx_logden(self, x):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            out = self.log_density(xg).sum()
            grad, = torch.autograd.grad(out, xg)
        return grad

    # used on log_density() to verify that dx_logden is right
    def test(self, x):
        return jacrev(self.log_density)(x)

    # draw n samples from copula. Khoudraji's device has an exact stochastic
    # representation -- no rejection, no inversion: if (U1,V1) ~ C1 and
    # (U2,V2) ~ C2 independently, then
    #     U = max(U1^(1/(1-a1)), U2^(1/a1)),  V = max(V1^(1/(1-a2)), V2^(1/a2))
    # has CDF P(U<=u, V<=v) = C1(u^(1-a1), v^(1-a2)) * C2(u^a1, v^a2) = K,
    # since the max splits into the two independent events.
    def sample_u(self, n_samples):
        a1, a2 = self.shape
        s1 = np.asarray(self.base1.sample_u(n_samples), dtype=float)
        s2 = np.asarray(self.base2.sample_u(n_samples), dtype=float)
        out = np.empty_like(s1)
        for i, a in enumerate((a1, a2)):
            # a = 0 kills the second stream (x^inf -> 0 for x < 1) and a = 1
            # the first, which is exactly the degenerate collapse to C1 / C2
            first = s1[:, i] ** (1.0 / (1.0 - a)) if a < 1 else np.zeros(n_samples)
            second = s2[:, i] ** (1.0 / a) if a > 0 else np.zeros(n_samples)
            out[:, i] = np.maximum(first, second)
        return out

    # draw n samples from f
    def sample_x(self, n_samples):
        sample = self.sample_u(n_samples)
        return norm.ppf(sample, loc=0, scale=self.sigma)


class GluedCopula(nn.Module):
    """Siburg-Stoimenov gluing: two copulas squeezed into complementary strips.

    Cut [0,1]^2 along the line {u_index = theta} and paste a scaled copy of
    C1 into the strip on one side of it and of C2 into the other. Written for
    index = 0 (a vertical cut), the copula is

        C(u, v) = theta * C1(u/theta, v)                          u <= theta
                = (1-theta) * C2((u-theta)/(1-theta), v) + theta*v  u >= theta

    (Siburg & Stoimenov, "Gluing Copulas", Comm. Statist. 37(19), eqs. 3-4;
    index = 1 is the same construction with the two coordinates swapped, a
    horizontal cut along {v = theta}.) The `theta*v` in the second branch is
    the paper's `theta*C1(1, v)`, which is `theta*v` since C1 is a copula.

    Density: the mass factor exactly cancels the Jacobian of the rescaling,
    so there is NO prefactor left -- the density is a plain piecewise copy,

        c(u, v) = c1(u/theta, v)  on the left strip,
                  c2((u-theta)/(1-theta), v)  on the right one,

    which is also the quickest way to see the marginals stay uniform:
    integrating over v gives 1 on each strip separately (each c_k has uniform
    marginals), and integrating over u gives theta*1 + (1-theta)*1 = 1.

    What makes this construction different from every other wrapper in this
    file is that the glued copula is *not* smooth: the density JUMPS across
    the seam, and each strip carries the base copula's whole u=0 / u=1 edge
    behaviour squeezed up against it, so c blows up (or vanishes) along the
    seam wherever the base does at its own edge. `MixtureCopula`,
    `MirroredCopula` and `KhoudrajiCopula` all leave the singularities in the
    corners, where the rest of the machinery expects them; this one puts a
    singular LINE through the middle of the square. Two consequences, both
    measured:

      * The fade script must take its CDF route for a glued copula (it does
        -- `copula_cdf_u` has a branch). Its pdf-quadrature fallback is off
        by up to 49% in a cell the seam crosses, against ~2e-4 elsewhere.
      * A glued endpoint is genuinely hard for the collocation-based trainer,
        whose loss reads log c and its gradient at grid points: over
        x in [-8 sigma, 8 sigma]^2, max|grad log f| runs ~1e3-4e5 across the
        glued pairs measured, against ~15-25 for the same bases unglued.
        Nothing here is infinite or NaN on that range, but the scale is two
        to four orders of magnitude larger, so expect the constraint penalty
        to work harder.

    Exactly ON the seam the density has no value to return -- its two
    one-sided limits differ, that being what "jump" means -- so the `<=` in
    `_branch_u` picks the left strip's limit by convention, matching the
    closed [0, theta] of the paper's eq. (3). Nothing downstream depends on
    the choice: it is one line of a measure-zero set, and the CDF, which is
    what the fade script actually reads, is continuous across it.

    Two properties from the paper worth knowing, both pinned in `self_check`
    in fade_distance.py:
      * gluing P with P gives P back, and nothing else glues to P (Prop. 3.1);
      * Spearman's rho and Kendall's tau transform the same way,
        rho_C = theta^2 rho_1 + (1-theta)^2 rho_2 (Thm. 3.2), so gluing
        always shrinks dependence towards independence -- C agrees with P on
        the whole seam by construction.

    base1, base2 : any objects with the `load_target` interface
        (`load_target`, `MixtureCopula`, `MirroredCopula`, `KhoudrajiCopula`,
        or another `GluedCopula`). base1 goes into the strip nearer 0.
    theta : cut position in (0,1); also the mass given to base1.
    index : 0 to cut along the first coordinate, 1 along the second.

    This class was implemented with Claude Code.
    """

    def __init__(self, base1, base2, theta=0.5, index=0):
        super().__init__()
        theta = float(theta)
        if not (0.0 < theta < 1.0):
            raise ValueError("the gluing point theta must lie strictly inside (0, 1)")
        if int(index) not in (0, 1):
            raise ValueError("index must be 0 or 1 (this is the bivariate construction)")
        if float(base1.sigma) != float(base2.sigma):
            raise ValueError("both components of a GluedCopula must share the same sigma")

        self.base1, self.base2 = base1, base2
        self.index = int(index)
        self.sigma = base1.sigma
        self.target = "GluedCopula"
        # unlike the other composite wrappers, `theta` here IS a real
        # parameter of this copula (where the cut goes), not a leftover of a
        # component, so it is exposed under the usual name
        self.theta = torch.tensor(theta)
        self.cut = theta  # plain float, for the arithmetic below

    def extra_repr(self):
        return f"theta={self.cut:g}, index={self.index}"

    # (u, v) -> the two bases' own coordinates: rescale the glued coordinate
    # by each strip's affine map and leave the other one alone.
    #
    # BOTH branches are returned everywhere, to be selected by torch.where at
    # the call site, and both are clamped into the open square: outside its
    # own strip a branch's raw argument leaves [0,1] (branch 2's
    # (u-theta)/(1-theta) is negative on the whole left strip), where a base
    # density is NaN or infinite -- discarded by `where` in the forward pass,
    # but still able to poison a gradient through it as 0 * inf.
    def _branch_u(self, u):
        i = self.index
        th = self.cut
        w, other = u[..., i], u[..., 1 - i]
        lower = w <= th
        w1 = (w / th).clamp(_UEPS, 1 - _UEPS)
        w2 = ((w - th) / (1 - th)).clamp(_UEPS, 1 - _UEPS)
        return lower, self._restack(w1, other), self._restack(w2, other)

    # put a (glued coordinate, other coordinate) pair back in input order
    def _restack(self, glued, other):
        pair = (glued, other) if self.index == 0 else (other, glued)
        return torch.stack(pair, dim=-1)

    # copula density [0,1]^2 -> R^+: a piecewise copy of the two bases',
    # with no scale factor (see the class docstring)
    def density_u(self, u):
        lower, u1, u2 = self._branch_u(u)
        return torch.where(lower, self.base1.density_u(u1), self.base2.density_u(u2))

    def log_density_u(self, u):
        lower, u1, u2 = self._branch_u(u)
        return torch.where(lower, self.base1.log_density_u(u1),
                           self.base2.log_density_u(u2))

    # joint CDF: eqs. (3)-(4) of the paper
    def cdf_u(self, u):
        lower, u1, u2 = self._branch_u(u)
        th, other = self.cut, u[..., 1 - self.index]
        return torch.where(lower,
                           th * self.base1.cdf_u(u1),
                           (1 - th) * self.base2.cdf_u(u2) + th * other)

    # dC/du_coord. Differentiating (3)-(4) w.r.t. the GLUED coordinate, the
    # 1/scale from the chain rule cancels the scale in front, so that
    # derivative is the base h-function unchanged; w.r.t. the other
    # coordinate nothing is rescaled, so the prefactors survive (and the
    # theta*v term of (4) contributes its theta).
    def _hfunc(self, u, coord):
        lower, u1, u2 = self._branch_u(u)
        name = "hfunc_u" if coord == 0 else "hfunc_v"
        h1, h2 = getattr(self.base1, name)(u1), getattr(self.base2, name)(u2)
        if coord == self.index:
            return torch.where(lower, h1, h2)
        th = self.cut
        return torch.where(lower, th * h1, (1 - th) * h2 + th)

    def hfunc_u(self, u):
        return self._hfunc(u, 0)

    def hfunc_v(self, u):
        return self._hfunc(u, 1)

    def density_x(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        return self.density_u(normal2.cdf(x)) * torch.exp(
            normal2.log_prob(x[..., 0]) + normal2.log_prob(x[..., 1]))

    def log_density(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        return (self.log_density_u(normal2.cdf(x))
                + normal2.log_prob(x[..., 0]) + normal2.log_prob(x[..., 1]))

    # Gradient of log_density(), hand-derived (unlike KhoudrajiCopula's, the
    # chain rule here is short enough to be worth doing in closed form).
    #
    # On a strip, log f(x) = log c_k(u') + log phi(x_0) + log phi(x_1) with
    # u'_i = (u_i - a_k)/s_k and u'_j = u_j, so with g = base_k.dx_logden(x')
    # -- the base's own x-space score at the rescaled point x' = Phi^-1(u') --
    # and d/du' log c_k = (g + x'/sigma^2) / phi(x') read off from the base's
    # own Sklar lift:
    #
    #   d/dx_j log f = g_j                                    (j != index)
    #   d/dx_i log f = (g_i + x'_i/sigma^2)/s_k * phi(x_i)/phi(x'_i)
    #                  - x_i/sigma^2                          (i  = index)
    #
    # The untouched coordinate passes through as g_j exactly, and note that
    # x'_j is taken as x_j directly rather than as icdf(cdf(x_j)): that round
    # trip returns +inf once cdf(x_j) rounds to 1.0, which happens well
    # inside the domain the trainer evaluates on.
    #
    # The density ratio phi(x_i)/phi(x'_i) is evaluated in the exponent. It
    # genuinely blows up as x_i approaches the seam from either side -- that
    # is the singular line described in the class docstring, not a numerical
    # artefact; the _UEPS clamp in `_branch_u` is what keeps it finite.
    def dx_logden(self, x):
        normal2 = torch.distributions.Normal(0.0, self.sigma)
        i, th, s2 = self.index, self.cut, self.sigma ** 2
        u = normal2.cdf(x)
        lower, u1, u2 = self._branch_u(u)
        xi, xo = x[..., i], x[..., 1 - i]

        out = torch.zeros_like(x)
        for uk, scale, base, mask in ((u1, th, self.base1, lower),
                                      (u2, 1 - th, self.base2, ~lower)):
            xk = normal2.icdf(uk[..., i])                      # x'_i
            g = base.dx_logden(self._restack(xk, xo))
            ratio = torch.exp((xk ** 2 - xi ** 2) / (2 * s2))  # phi(x_i)/phi(x'_i)
            glued = (g[..., i] + xk / s2) / scale * ratio - xi / s2
            out[..., i] = torch.where(mask, glued, out[..., i])
            out[..., 1 - i] = torch.where(mask, g[..., 1 - i], out[..., 1 - i])
        return out

    # used on log_density() to verify that dx_logden is right
    def test(self, x):
        return jacrev(self.log_density)(x)

    # draw n samples from copula: the left strip holds exactly mass theta, so
    # split the draws by a Binomial(n, theta) and squeeze each base's sample
    # into its own strip -- exact, no rejection
    def sample_u(self, n_samples):
        i, th = self.index, self.cut
        n1 = int(np.random.binomial(n_samples, th))
        parts = []
        for base, n, lo, width in ((self.base1, n1, 0.0, th),
                                   (self.base2, n_samples - n1, th, 1 - th)):
            if n > 0:
                s = np.asarray(base.sample_u(n), dtype=float).copy()
                s[:, i] = lo + width * s[:, i]
                parts.append(s)
        sample = np.concatenate(parts, axis=0)
        np.random.shuffle(sample)
        return sample

    # draw n samples from f
    def sample_x(self, n_samples):
        sample = self.sample_u(n_samples)
        return norm.ppf(sample, loc=0, scale=self.sigma)
