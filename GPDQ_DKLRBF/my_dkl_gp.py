import torch
import torch.nn as nn

_SQRT3 = torch.math.sqrt(torch.tensor(3.0, dtype=torch.float32))
_SQRT5 = torch.math.sqrt(torch.tensor(5.0, dtype=torch.float32))


class FeatureNet(nn.Module):
    """Maps full state → d-dimensional GP embedding (tanh-bounded, compact range)."""
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class myDKLGP(nn.Module):
    """Deep Kernel Learning GP: FeatureNet(full state) → emb_dim → GP kernel.

    The feature network and GP hyperparameters are jointly optimised via MLL.
    Unlike myExactGP, x_train stores the *full* state (no dim slicing) so the
    NN can exploit all available information.
    """

    def __init__(self, params_dict: dict, dataset: dict, parent_alg: str, cuda: bool = False):
        super().__init__()
        self.parent_alg = parent_alg
        self.alg = 'dkl_gp'
        self.param_dict = params_dict
        self.env = params_dict['environment']
        self.state_dim = int(params_dict['state_dim'])

        # ── Environment-conditional defaults ────────────────────────────────────
        if 'antmaze' in self.env or 'maze2d' in self.env:
            self.kernel_fn = 'matern32'   # rough kernel suits wall-structured mazes
            self.emb_dim   = 4            # (x, y, sin-heading, cos-heading) implicitly
        elif 'halfcheetah' in self.env:
            self.kernel_fn = 'matern52'   # smooth physical map
            self.emb_dim   = 6            # enough to capture height + 5 joint factors
        else:
            self.kernel_fn = 'matern52'
            self.emb_dim   = 4
        self.kernel_fn = params_dict.get('kernel_fn', self.kernel_fn)
        self.emb_dim   = int(params_dict.get('emb_dim', self.emb_dim))

        # ── Dataset / sizing ────────────────────────────────────────────────────
        self.num_sample = dataset['arr_0']
        self.gp_training_size = params_dict['gp_num_sample']
        if self.gp_training_size > self.num_sample:
            self.gp_training_size = self.num_sample
        self.y_dim = int(params_dict['action_dim'])

        self.x_train_full = torch.tensor(dataset['observations'], dtype=torch.float32)
        self.y_train_full = torch.tensor(dataset['actions'],      dtype=torch.float32)
        # _org copies kept for getAlteredObservation resets (full-state, full-action)
        self.x_train_org = torch.tensor(self.x_train_full[:self.gp_training_size], dtype=torch.float32)
        self.y_train_org = torch.tensor(self.y_train_full[:self.gp_training_size], dtype=torch.float32)

        print('========== Create new DKL GP record and models!')
        self.mll_append = []

        # ── Feature network + GP hyperparameters ────────────────────────────────
        self.feature_net = FeatureNet(in_dim=self.state_dim, emb_dim=self.emb_dim)
        self.sigma_p     = torch.tensor(1.0, dtype=torch.float32)
        self.sigma_n     = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.ell         = nn.Parameter(torch.ones(1, self.emb_dim, dtype=torch.float32), requires_grad=True)

        # Single joint optimiser for NN + GP params
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-03)

        # ── Training data (full state stored; embedding computed on the fly) ────
        _start = 0
        self.x_train = torch.tensor(
            self.x_train_full[_start:_start + self.gp_training_size], dtype=torch.float32)
        self.y_train = torch.tensor(
            self.y_train_full[_start:_start + self.gp_training_size], dtype=torch.float32)

        # Cache embedded training data + kernel (updated after each training run)
        with torch.no_grad():
            self.z_train = self.feature_net(self.x_train)
        self.K = self.kernel(self.z_train, self.z_train, noise=True)

        print(f'DKL GP | kernel: {self.kernel_fn} | emb_dim: {self.emb_dim} | '
              f'state_dim: {self.state_dim} | n_train: {len(self.x_train)}')

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _ensure_tensor(self, X):
        return X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)

    def _ard_r(self, Z_1, Z_2):
        return torch.cdist(Z_1 / self.ell, Z_2 / self.ell)

    def _noise_term(self, n):
        noise_var = torch.clamp(self.sigma_n ** 2, min=1e-4)
        return noise_var * torch.eye(n, dtype=torch.float32, device=self.sigma_n.device)

    # ── Kernel functions (operate on embeddings) ─────────────────────────────────

    def rbfKernel(self, Z_1, Z_2, noise=False):
        r = self._ard_r(Z_1, Z_2)
        k = (self.sigma_p ** 2) * torch.exp(-(r ** 2) / 2.0)
        if noise:
            k = k + self._noise_term(len(Z_1))
        return k

    def maternKernel(self, Z_1, Z_2, noise=False):
        r  = self._ard_r(Z_1, Z_2)
        sr = _SQRT3 * r
        k  = (self.sigma_p ** 2) * (1.0 + sr) * torch.exp(-sr)
        if noise:
            k = k + self._noise_term(len(Z_1))
        return k

    def matern52Kernel(self, Z_1, Z_2, noise=False):
        r  = self._ard_r(Z_1, Z_2)
        sr = _SQRT5 * r
        k  = (self.sigma_p ** 2) * (1.0 + sr + (5.0 / 3.0) * r ** 2) * torch.exp(-sr)
        if noise:
            k = k + self._noise_term(len(Z_1))
        return k

    def kernel(self, Z_1, Z_2, noise=False):
        if self.kernel_fn == 'rbf':
            return self.rbfKernel(Z_1, Z_2, noise)
        elif self.kernel_fn == 'matern32':
            return self.maternKernel(Z_1, Z_2, noise)
        elif self.kernel_fn == 'matern52':
            return self.matern52Kernel(Z_1, Z_2, noise)
        else:
            raise ValueError(f'Unknown kernel_fn: {self.kernel_fn!r}')

    # ── GP inference ─────────────────────────────────────────────────────────────

    def predict(self, X_s):
        X_s    = self._ensure_tensor(X_s).view(-1, self.state_dim)
        with torch.no_grad():
            z_test = self.feature_net(X_s)
            _k_s   = self.kernel(Z_1=self.z_train, Z_2=z_test, noise=False)
            _k_ss  = self.kernel(Z_1=z_test,       Z_2=z_test, noise=False)
            _L     = torch.linalg.cholesky(self.K, upper=False)
            _alpha = torch.linalg.solve_triangular(
                        _L.T,
                        torch.linalg.solve_triangular(_L, self.y_train, upper=False),
                        upper=True)
            _mean = _k_s.T @ _alpha
            _v    = torch.linalg.solve_triangular(_L, _k_s, upper=False)
            _var  = torch.clamp(_k_ss - _v.T @ _v, min=0.0)
        return _mean, _var

    # ── MLL training ─────────────────────────────────────────────────────────────

    def myTraining(self, total_epoch: int, ft: bool = False):
        _batch_size     = self.gp_training_size
        _gradient_steps = self.gp_training_size // _batch_size

        _training_record = 0
        while _training_record < total_epoch:
            for g in range(_gradient_steps):
                _batch_x = self.x_train[g * _batch_size: _batch_size + g * _batch_size]
                _batch_y = self.y_train[g * _batch_size: _batch_size + g * _batch_size]

                self.optimizer.zero_grad()
                # Gradient flows through both FeatureNet and GP hyperparameters
                z_batch = self.feature_net(_batch_x)
                _k  = self.kernel(Z_1=z_batch, Z_2=z_batch, noise=True)
                _L  = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(
                            _L.T,
                            torch.linalg.solve_triangular(_L, _batch_y, upper=False),
                            upper=True)
                _mll = (-0.5 * _batch_y.T @ _alpha) \
                       - torch.sum(torch.log(torch.diagonal(_L))) \
                       - (_batch_size * torch.log(torch.tensor(2 * torch.pi)) / 2)
                _mll = -_mll.mean()
                _mll.backward(retain_graph=True)
                self.optimizer.step()
                self.mll_append.append(_mll.item())

            _training_record += 1

        # Refresh cached embedding + K with the final trained feature net
        with torch.no_grad():
            self.z_train = self.feature_net(self.x_train)
            self.K = self.kernel(self.z_train, self.z_train, noise=True)

        return _mll.item()

    def recordSaving(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),   # includes FeatureNet weights + GP params
            'loss_append': self.mll_append,
            'x_train': self.x_train,
            'y_train': self.y_train,
            'z_train': self.z_train,
        }, path)
