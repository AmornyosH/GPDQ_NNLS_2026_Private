import os
import torch
import numpy as np
from time import time
from utility import my_NN, my_utils

_SQRT3 = torch.math.sqrt(torch.tensor(3.0, dtype=torch.float32))
_SQRT5 = torch.math.sqrt(torch.tensor(5.0, dtype=torch.float32))

# Kept for backward-compat with any code that imports this name
TORCH_SQRT_3 = _SQRT3


class myExactGP(torch.nn.Module):
    def __init__(self, params_dict: dict, dataset: dict, parent_alg: str, cuda: bool = False):
        super().__init__()
        self.parent_alg = parent_alg
        self.alg = 'exact_gp'
        self.param_dict = params_dict
        self.env = self.param_dict['environment']

        # ── Environment-conditional kernel and input-dim selection ──────────────
        if 'antmaze' in self.env or 'maze2d' in self.env:
            self.kernel_fn = 'matern32'   # rough kernel: walls/turns need fast correlation decay
            self.x_dim = 2                # (x, y) position — complete spatial prior
        elif 'halfcheetah' in self.env:
            self.kernel_fn = 'matern52'   # smooth physical map; try 'rbf_periodic' for gait cycle
            self.x_dim = 3                # (z-height, root-angle, angular-velocity) breaks phase degeneracy
        else:
            self.kernel_fn = 'matern52'   # safe default for other MuJoCo locomotion tasks
            self.x_dim = 2
        # Allow params_dict override, e.g. {'kernel_fn': 'rbf_periodic', 'gp_x_dim': 2}
        self.kernel_fn = params_dict.get('kernel_fn', self.kernel_fn)
        self.x_dim = int(params_dict.get('gp_x_dim', self.x_dim))

        # ── Dataset / sizing ────────────────────────────────────────────────────
        _best_dataset = dataset
        self.num_sample = _best_dataset['arr_0']
        self.gp_training_size = self.param_dict['gp_num_sample']
        if self.gp_training_size > self.num_sample:
            self.gp_training_size = self.num_sample
        self.y_dim = int(self.param_dict['action_dim'])
        self.x_train_full = torch.tensor(_best_dataset['observations'], dtype=torch.float32)
        self.y_train_full = torch.tensor(_best_dataset['actions'],      dtype=torch.float32)
        self.x_train_org  = torch.tensor(self.x_train_full[:self.gp_training_size], dtype=torch.float32)
        self.y_train_org  = torch.tensor(self.y_train_full[:self.gp_training_size], dtype=torch.float32)

        print('========== Create new GP record and models!')
        self.mll_append = []

        # ── Learnable kernel hyperparameters ────────────────────────────────────
        # sigma_p: signal variance (fixed to 1.0; add nn.Parameter here if you want to learn it)
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)
        # Noise / jitter  (floored to 1e-4 in kernel to prevent near-singular K)
        self.sigma_n  = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        # ARD length-scales — one per input dim (already active in original code)
        self.ell      = torch.nn.Parameter(torch.ones(1, self.x_dim, dtype=torch.float32), requires_grad=True)
        # Periodic-kernel extras (only used when kernel_fn == 'rbf_periodic')
        self.log_period = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        self.ell_per    = torch.nn.Parameter(torch.ones(1,  dtype=torch.float32), requires_grad=True)

        # Optimizer constructed AFTER all nn.Parameters are registered
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-03)

        self.cuda() if cuda else ...

        # ── Training data matrices ───────────────────────────────────────────────
        _start = 0
        self.x_train = torch.tensor(
            self.x_train_full[_start:_start + self.gp_training_size, 0:self.x_dim], dtype=torch.float32)
        self.y_train = torch.tensor(
            self.y_train_full[_start:_start + self.gp_training_size], dtype=torch.float32)

        print(f'GP kernel: {self.kernel_fn}  |  x_dim: {self.x_dim}  |  training size: {len(self.x_train)}')
        self.K = self.kernel(X_1=self.x_train, X_2=self.x_train, noise=True)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _ensure_tensor(self, X):
        return X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)

    def _ard_r(self, X_1, X_2):
        """ARD Euclidean distance: ||x_1/ell - x_2/ell||"""
        return torch.cdist(X_1 / self.ell, X_2 / self.ell)

    def _noise_term(self, n):
        """Jitter + noise diagonal; floored so K stays positive-definite."""
        noise_var = torch.clamp(self.sigma_n ** 2, min=1e-4)
        return noise_var * torch.eye(n, dtype=torch.float32, device=self.sigma_n.device)

    # ── Kernel functions ─────────────────────────────────────────────────────────

    def rbfKernel(self, X_1, X_2, noise=False):
        """Squared Exponential (RBF) with ARD."""
        X_1 = self._ensure_tensor(X_1)
        X_2 = self._ensure_tensor(X_2)
        r = self._ard_r(X_1, X_2)
        k = (self.sigma_p ** 2) * torch.exp(-(r ** 2) / 2.0)
        if noise:
            k = k + self._noise_term(len(X_1))
        return k

    def maternKernel(self, X_1, X_2, noise=False):
        """Matérn 3/2 with ARD — recommended for AntMaze (wall-aware, fast decay)."""
        X_1 = self._ensure_tensor(X_1)
        X_2 = self._ensure_tensor(X_2)
        r  = self._ard_r(X_1, X_2)
        sr = _SQRT3 * r
        k  = (self.sigma_p ** 2) * (1.0 + sr) * torch.exp(-sr)
        if noise:
            k = k + self._noise_term(len(X_1))
        return k

    def matern52Kernel(self, X_1, X_2, noise=False):
        """Matérn 5/2 with ARD — recommended for HalfCheetah (smooth physical map)."""
        X_1 = self._ensure_tensor(X_1)
        X_2 = self._ensure_tensor(X_2)
        r  = self._ard_r(X_1, X_2)
        sr = _SQRT5 * r
        k  = (self.sigma_p ** 2) * (1.0 + sr + (5.0 / 3.0) * r ** 2) * torch.exp(-sr)
        if noise:
            k = k + self._noise_term(len(X_1))
        return k

    def rbfPeriodicKernel(self, X_1, X_2, noise=False):
        """RBF(dim0) × Periodic(dim1) — for HalfCheetah gait cycle.
        dim0 = root z-height (smooth amplitude), dim1 = root angle (periodic gait phase).
        """
        X_1 = self._ensure_tensor(X_1)
        X_2 = self._ensure_tensor(X_2)
        # RBF on z-height (dim 0)
        z1 = X_1[:, 0:1] / self.ell[:, 0:1]
        z2 = X_2[:, 0:1] / self.ell[:, 0:1]
        k_rbf = torch.exp(-(torch.cdist(z1, z2) ** 2) / 2.0)
        # Periodic on root-angle (dim 1)
        a1 = X_1[:, 1:2]
        a2 = X_2[:, 1:2]
        p  = torch.exp(self.log_period)                        # learnable period
        d  = torch.cdist(a1, a2)                               # |angle_i − angle_j|
        k_per = torch.exp(-2.0 * torch.sin(torch.pi * d / p) ** 2 / (self.ell_per ** 2))
        k = (self.sigma_p ** 2) * k_rbf * k_per
        if noise:
            k = k + self._noise_term(len(X_1))
        return k

    def addKernel(self, X_1, X_2, noise=False):
        """RBF (long-range drift) + Matérn-3/2 (local roughness) — composite for AntMaze."""
        X_1 = self._ensure_tensor(X_1)
        X_2 = self._ensure_tensor(X_2)
        # Global smooth trend: RBF with large effective length-scale (ell * 3)
        r_global = torch.cdist(X_1 / (self.ell * 3.0), X_2 / (self.ell * 3.0))
        k_rbf    = 0.5 * (self.sigma_p ** 2) * torch.exp(-(r_global ** 2) / 2.0)
        # Local rough structure: Matérn-3/2 with learned ell
        r  = self._ard_r(X_1, X_2)
        sr = _SQRT3 * r
        k_m32 = 0.5 * (self.sigma_p ** 2) * (1.0 + sr) * torch.exp(-sr)
        k = k_rbf + k_m32
        if noise:
            k = k + self._noise_term(len(X_1))
        return k

    def kernel(self, X_1, X_2, noise=False):
        """Dispatcher — routes to the env-selected kernel function."""
        if self.kernel_fn == 'rbf':
            return self.rbfKernel(X_1, X_2, noise)
        elif self.kernel_fn == 'matern32':
            return self.maternKernel(X_1, X_2, noise)
        elif self.kernel_fn == 'matern52':
            return self.matern52Kernel(X_1, X_2, noise)
        elif self.kernel_fn == 'rbf_periodic':
            return self.rbfPeriodicKernel(X_1, X_2, noise)
        elif self.kernel_fn == 'rbf_matern32':
            return self.addKernel(X_1, X_2, noise)
        else:
            raise ValueError(f'Unknown kernel_fn: {self.kernel_fn!r}')

    # ── GP inference ────────────────────────────────────────────────────────────

    def predict(self, X_s):
        x_test = X_s[:, 0:self.x_dim].view(-1, self.x_dim)

        with torch.no_grad():
            _k_s  = self.kernel(X_1=self.x_train, X_2=x_test,  noise=False)
            _k_ss = self.kernel(X_1=x_test,        X_2=x_test,  noise=False)
            _L     = torch.linalg.cholesky(self.K, upper=False)
            _alpha = torch.linalg.solve_triangular(
                        _L.T,
                        torch.linalg.solve_triangular(_L, self.y_train, upper=False),
                        upper=True)
            _mean = _k_s.T @ _alpha
            _v    = torch.linalg.solve_triangular(_L, _k_s, upper=False)
            _var  = _k_ss - _v.T @ _v
            # Clamp to prevent negative-variance numerical artefacts feeding the guidance gate
            _var  = torch.clamp(_var, min=0.0)

        return _mean, _var

    # ── MLL training ────────────────────────────────────────────────────────────

    def myTraining(self, total_epoch: int, ft: bool = False):
        _batch_size     = self.gp_training_size
        _gradient_steps = self.gp_training_size // _batch_size

        _training_record = 0
        while _training_record < total_epoch:
            for g in range(_gradient_steps):
                _batch_x = self.x_train[g * _batch_size: _batch_size + g * _batch_size]
                _batch_y = self.y_train[g * _batch_size: _batch_size + g * _batch_size]

                self.optimizer.zero_grad()
                _k  = self.kernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                _L  = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(
                            _L.T,
                            torch.linalg.solve_triangular(_L, _batch_y, upper=False),
                            upper=True)
                # Multi-output MLL (independent GPs per action dim, shared kernel K):
                #   data-fit = -½ tr(Yᵀ K⁻¹ Y)  [trace, not .mean() — avoids off-diagonal terms]
                #   log-det  = -action_dim · Σ log diag(L)
                _data_fit = -0.5 * torch.trace(_batch_y.T @ _alpha)
                _logdet   = -self.y_dim * torch.sum(torch.log(torch.diagonal(_L)))
                _const    = -(self.y_dim * _batch_size / 2.0) * torch.log(torch.tensor(2 * torch.pi))
                _mll = -(_data_fit + _logdet + _const)
                _mll.backward()   # no retain_graph: graph rebuilt each step
                self.optimizer.step()
                self.mll_append.append(_mll.item())

            _training_record += 1

        # Recompute cached K with updated hyperparameters
        self.K = self.kernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        return _mll.item()

    def recordSaving(self, path: str):
        torch.save({'state_dict': self.state_dict(),
                    'loss_append': self.mll_append,
                    'x_train': self.x_train,
                    'y_train': self.y_train}, path)
