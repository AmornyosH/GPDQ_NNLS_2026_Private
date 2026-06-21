import torch
import torch.nn as nn

_SQRT3 = torch.math.sqrt(torch.tensor(3.0, dtype=torch.float32))
_SQRT5 = torch.math.sqrt(torch.tensor(5.0, dtype=torch.float32))


class FeatureNet(nn.Module):
    """Maps full state → d-dimensional GP embedding.
    LayerNorm on the output forces unit per-dimension variance, which is the
    single most effective defence against embedding collapse (all states → same
    point → singular K).  Tanh on hidden layers keeps gradients stable.
    """
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, emb_dim),
            nn.LayerNorm(emb_dim),   # unit-variance embedding → well-conditioned K
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class myDKLGP(nn.Module):
    """Deep Kernel Learning GP: FeatureNet(full state) → emb_dim → GP kernel.

    Jointly optimises FeatureNet weights + GP hyperparameters via MLL so the
    learned metric reflects action-relevant structure rather than a fixed
    coordinate slice.
    """

    def __init__(self, params_dict: dict, dataset: dict, parent_alg: str, cuda: bool = False):
        super().__init__()
        self.parent_alg = parent_alg
        self.alg = 'dkl_gp'
        self.param_dict = params_dict
        self.env = params_dict['environment']
        self.state_dim = int(params_dict['state_dim'])
        self.y_dim     = int(params_dict['action_dim'])

        # ── Environment-conditional defaults ────────────────────────────────────
        if 'antmaze' in self.env or 'maze2d' in self.env:
            self.kernel_fn = 'matern32'
            self.emb_dim   = 4
        elif 'halfcheetah' in self.env:
            self.kernel_fn = 'matern52'
            self.emb_dim   = 6
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

        self.x_train_full = torch.tensor(dataset['observations'], dtype=torch.float32)
        self.y_train_full = torch.tensor(dataset['actions'],      dtype=torch.float32)
        # _org copies kept for getAlteredObservation resets (must persist in checkpoint)
        self.x_train_org = torch.tensor(self.x_train_full[:self.gp_training_size], dtype=torch.float32)
        self.y_train_org = torch.tensor(self.y_train_full[:self.gp_training_size], dtype=torch.float32)

        print('========== Create new DKL GP record and models!')
        self.mll_append = []

        # ── Feature network + learnable GP hyperparameters ───────────────────────
        self.feature_net  = FeatureNet(in_dim=self.state_dim, emb_dim=self.emb_dim)
        # log_sigma_p: learnable signal std (log-parameterised → always positive)
        self.log_sigma_p  = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        # sigma_n: noise/jitter (floored at 1e-3 in _noise_term)
        self.sigma_n      = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        # ARD length-scales in embedding space
        self.ell          = nn.Parameter(torch.ones(1, self.emb_dim, dtype=torch.float32), requires_grad=True)

        # weight_decay adds L2 regularisation on FeatureNet — limits metric distortion
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-03, weight_decay=1e-4)

        # ── Training data (full state — no x_dim slicing) ────────────────────────
        _start = 0
        self.x_train = torch.tensor(
            self.x_train_full[_start:_start + self.gp_training_size], dtype=torch.float32)
        self.y_train = torch.tensor(
            self.y_train_full[_start:_start + self.gp_training_size], dtype=torch.float32)

        # Cache: z_train = φ(x_train), K = kernel(z_train, z_train)
        # Rebuilt after every myTraining call.  _cache_stale guards against accidental
        # de-sync if training cadence changes.
        with torch.no_grad():
            self.z_train = self.feature_net(self.x_train)
        self.K = self.kernel(self.z_train, self.z_train, noise=True)
        self._cache_stale = False

        print(f'DKL GP | kernel: {self.kernel_fn} | emb_dim: {self.emb_dim} | '
              f'state_dim: {self.state_dim} | n_train: {len(self.x_train)}')

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _ensure_tensor(self, X):
        return X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)

    def _ard_r(self, Z_1, Z_2):
        return torch.cdist(Z_1 / self.ell, Z_2 / self.ell)

    def _noise_term(self, n):
        # Floor at 1e-3 (raised from 1e-4): 1000×1000 Matérn over learned embeddings
        # needs a higher jitter floor than a fixed-coord kernel.
        noise_var = torch.clamp(self.sigma_n ** 2, min=1e-3)
        return noise_var * torch.eye(n, dtype=torch.float32, device=self.sigma_n.device)

    def _sigma_p_sq(self):
        return torch.exp(self.log_sigma_p) ** 2

    # ── Kernel functions (operate on embeddings) ─────────────────────────────────

    def rbfKernel(self, Z_1, Z_2, noise=False):
        r = self._ard_r(Z_1, Z_2)
        k = self._sigma_p_sq() * torch.exp(-(r ** 2) / 2.0)
        if noise:
            k = k + self._noise_term(len(Z_1))
        return k

    def maternKernel(self, Z_1, Z_2, noise=False):
        r  = self._ard_r(Z_1, Z_2)
        sr = _SQRT3 * r
        k  = self._sigma_p_sq() * (1.0 + sr) * torch.exp(-sr)
        if noise:
            k = k + self._noise_term(len(Z_1))
        return k

    def matern52Kernel(self, Z_1, Z_2, noise=False):
        r  = self._ard_r(Z_1, Z_2)
        sr = _SQRT5 * r
        k  = self._sigma_p_sq() * (1.0 + sr + (5.0 / 3.0) * r ** 2) * torch.exp(-sr)
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
        X_s = self._ensure_tensor(X_s).view(-1, self.state_dim)
        with torch.no_grad():
            # Rebuild cache if feature_net was updated since last myTraining refresh
            if self._cache_stale:
                self.z_train = self.feature_net(self.x_train)
                self.K = self.kernel(self.z_train, self.z_train, noise=True)
                self._cache_stale = False
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
                z_batch = self.feature_net(_batch_x)

                _k  = self.kernel(Z_1=z_batch, Z_2=z_batch, noise=True)
                _L  = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(
                            _L.T,
                            torch.linalg.solve_triangular(_L, _batch_y, upper=False),
                            upper=True)

                # Multi-output MLL (independent GPs per action dim, shared kernel K):
                #   data-fit  = -½ tr(Yᵀ K⁻¹ Y)  [trace sums per-output terms only]
                #   log-det   = -action_dim · Σ log diag(L)
                #   constant  = -(action_dim · N / 2) · log 2π
                _data_fit = -0.5 * torch.trace(_batch_y.T @ _alpha)
                _logdet   = -self.y_dim * torch.sum(torch.log(torch.diagonal(_L)))
                _const    = -(self.y_dim * _batch_size / 2.0) * torch.log(torch.tensor(2 * torch.pi))

                # Soft anti-collapse penalty: penalise near-zero per-dim spread in the embedding
                _spread_pen = torch.relu(0.1 - z_batch.std(dim=0)).sum()

                _mll = -(_data_fit + _logdet + _const) + 1e-2 * _spread_pen
                _mll.backward()   # no retain_graph: graph is rebuilt each step
                self.optimizer.step()
                self._cache_stale = True   # feature_net params just changed
                self.mll_append.append(_mll.item())

            _training_record += 1

        # Refresh cache with the final trained feature net
        with torch.no_grad():
            self.z_train = self.feature_net(self.x_train)
            self.K = self.kernel(self.z_train, self.z_train, noise=True)
        self._cache_stale = False

        # Diagnostic: log min per-dim std so collapse is detectable
        _emb_std = self.z_train.std(dim=0).min().item()
        print(f'  [DKL] MLL loss: {_mll.item():.4f}  |  min emb std: {_emb_std:.4f}')

        return _mll.item()

    def recordSaving(self, path: str):
        torch.save({
            'state_dict':   self.state_dict(),   # FeatureNet + log_sigma_p + sigma_n + ell
            'loss_append':  self.mll_append,
            'x_train':      self.x_train,
            'y_train':      self.y_train,
            'z_train':      self.z_train,
            # org copies are positionally aligned with GP training set; must survive reload
            'x_train_org':  self.x_train_org,
            'y_train_org':  self.y_train_org,
        }, path)
