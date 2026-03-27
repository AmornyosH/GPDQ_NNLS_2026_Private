"""
Multi-Output Stochastic Variational Gaussian Process (SVGP)
===========================================================
Output dimension: P = 6  (independent GPs, one per output)

Key change from single-output version:
  - q_mean    : (m,)    →  (m, P)
  - q_log_diag: (m,)    →  (m, P)
  - q_lower   : (m, m)  →  (m, m, P)
  - predict() returns (n, P) mean and (n, P) variance
  - KL and ELBO sum over all P outputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1.  KERNEL
# ─────────────────────────────────────────────

class RBFKernel(nn.Module):
    """
    k(x, x') = σ² · exp(−½ · ‖x − x'‖² / l²)
    One set of hyperparameters shared across all outputs.
    (You could also have per-output kernels — just replicate P times.)
    """

    def __init__(self, lengthscale: float = 1.0, outputscale: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(lengthscale)))
        self.log_outputscale = nn.Parameter(torch.tensor(math.log(outputscale)))

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    @property
    def outputscale(self):
        return self.log_outputscale.exp()

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Args:  X1 (n, d),  X2 (m, d)
        Returns: K (n, m)
        """
        X1s = X1 / self.lengthscale
        X2s = X2 / self.lengthscale
        sq_dists = (
            (X1s ** 2).sum(-1, keepdim=True)
            + (X2s ** 2).sum(-1).unsqueeze(0)
            - 2.0 * X1s @ X2s.T
        ).clamp(min=0.0)
        return self.outputscale ** 2 * torch.exp(-0.5 * sq_dists)


# ─────────────────────────────────────────────
# 2.  MULTI-OUTPUT SVGP
# ─────────────────────────────────────────────

class SVGP(nn.Module):
    """
    Multi-output SVGP with P independent output dimensions.

    Each output p has its own variational distribution:
        q_p(u) = N(m_p, S_p)   where S_p = L_p L_pᵀ

    The kernel (and inducing locations Z) are shared across outputs.
    This is the "intrinsic model of coregionalization" with rank-0
    cross-output covariance — i.e., outputs are independent given Z.

    Args:
        kernel:        RBFKernel
        inducing_pts:  (m, d) initial inducing locations
        n_outputs:     P, number of output dimensions (default 6)
        noise:         initial observation noise (shared across outputs)
        jitter:        numerical stability diagonal
    """

    def __init__(
        self,
        kernel: nn.Module,
        inducing_pts: torch.Tensor,
        n_outputs: int = 6,
        noise: float = 0.1,
        jitter: float = 1e-6,
    ):
        super().__init__()
        self.kernel   = kernel
        self.jitter   = jitter
        self.P        = n_outputs

        m, d = inducing_pts.shape

        # Inducing locations — shared across outputs
        self.Z = nn.Parameter(inducing_pts.clone())                  # (m, d)

        # ── Variational parameters — one set per output ──────────
        # Mean:  q_mean[:, p]  is the variational mean for output p
        self.q_mean     = nn.Parameter(torch.zeros(m, n_outputs))   # (m, P)

        # Cholesky of S_p stored as:
        #   diagonal part (log, for positivity): q_log_diag[:, p]
        #   strictly lower triangular part:      q_lower[:, :, p]
        self.q_log_diag = nn.Parameter(torch.zeros(m, n_outputs))   # (m, P)
        self.q_lower    = nn.Parameter(torch.zeros(m, m, n_outputs)) # (m, m, P)

        # Noise — one per output for flexibility
        self.log_noise  = nn.Parameter(
            torch.full((n_outputs,), math.log(noise))
        )                                                            # (P,)

    # ── helpers ──────────────────────────────

    @property
    def noise(self):
        return self.log_noise.exp()                                  # (P,)

    def _jitter_eye(self, n: int, device) -> torch.Tensor:
        return self.jitter * torch.eye(n, device=device)

    def _chol_S(self, p: int) -> torch.Tensor:
        """
        Build lower-triangular Cholesky factor L_p for output p.
        S_p = L_p L_pᵀ
        """
        L = torch.tril(self.q_lower[:, :, p], diagonal=-1)
        L = L + torch.diag(self.q_log_diag[:, p].exp())
        return L                                                     # (m, m)

    # ── KL divergence (summed over all outputs) ───────────────────

    def kl_divergence(self) -> torch.Tensor:
        """
        KL_total = Σ_p KL[ q_p(u) ‖ p(u) ]

        Each term:
        KL_p = ½ [ tr(Kzz⁻¹ S_p) + m_pᵀ Kzz⁻¹ m_p − m + log|Kzz| − log|S_p| ]

        Kzz is shared → compute its Cholesky once.
        """
        device = self.Z.device
        m = self.q_mean.shape[0]

        Kzz   = self.kernel(self.Z, self.Z) + self._jitter_eye(m, device)  # (m, m)
        L_kzz = torch.linalg.cholesky(Kzz)                                  # (m, m)
        log_det_Kzz = 2.0 * L_kzz.diagonal().log().sum()

        kl_total = torch.tensor(0.0, device=device)

        for p in range(self.P):
            L_q = self._chol_S(p)                                           # (m, m)
            S   = L_q @ L_q.T                                               # (m, m)

            mu_p = self.q_mean[:, p].unsqueeze(-1)                          # (m, 1)

            KzzInv_mu = torch.cholesky_solve(mu_p, L_kzz)                  # (m, 1)
            KzzInv_S  = torch.cholesky_solve(S, L_kzz)                     # (m, m)

            log_det_S = 2.0 * L_q.diagonal().log().sum()

            kl_p = 0.5 * (
                KzzInv_S.trace()
                + (mu_p.T @ KzzInv_mu).squeeze()
                - m
                + log_det_Kzz
                - log_det_S
            )
            kl_total = kl_total + kl_p

        return kl_total

    # ── predictive distribution ────────────────────────────────────

    def predict(self, X: torch.Tensor):
        """
        Compute q(f*) at test points X for all P outputs.

        Returns:
            mean : (n, P)   predictive mean
            var  : (n, P)   predictive marginal variance (no noise)
        """
        device = X.device
        n = X.shape[0]
        m = self.q_mean.shape[0]

        Kzz      = self.kernel(self.Z, self.Z) + self._jitter_eye(m, device)  # (m, m)
        Kxz      = self.kernel(X, self.Z)                                      # (n, m)
        Kxx_diag = self.kernel(X, X).diagonal()                                # (n,)

        L_kzz = torch.linalg.cholesky(Kzz)                                    # (m, m)

        # W = Kzz⁻¹ Kzxᵀ   (m, n)  — shared across outputs
        W = torch.cholesky_solve(Kxz.T, L_kzz)                               # (m, n)

        means = torch.zeros(n, self.P, device=device)
        vars_ = torch.zeros(n, self.P, device=device)

        for p in range(self.P):
            # α_p = Kzz⁻¹ m_p
            alpha_p = torch.cholesky_solve(
                self.q_mean[:, p].unsqueeze(-1), L_kzz
            ).squeeze(-1)                                                      # (m,)

            # Predictive mean for output p
            means[:, p] = Kxz @ alpha_p                                       # (n,)

            # Predictive variance for output p
            L_q = self._chol_S(p)                                             # (m, m)
            S   = L_q @ L_q.T                                                 # (m, m)

            KzzMinusS = Kzz - S                                               # (m, m)
            tr_term   = (W * (KzzMinusS @ W)).sum(dim=0)                     # (n,)

            vars_[:, p] = (Kxx_diag - tr_term).clamp(min=0.0)               # (n,)

        return means, vars_   # (n, P), (n, P)

    # ── ELBO ──────────────────────────────────────────────────────

    def elbo(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_total: int,
    ) -> torch.Tensor:
        """
        Mini-batch ELBO for multi-output regression.

        Args:
            X:       (B, d)   input batch
            y:       (B, P)   target batch  — all P outputs
            n_total: N        full dataset size (for scaling)

        ELBO = (N/B) · Σ_p Σ_i E_q[log N(y_ip | f_ip, σ²_p)]
               − Σ_p KL_p
        """
        B = X.shape[0]
        mean, var = self.predict(X)              # (B, P), (B, P)
        noise     = self.noise                   # (P,)

        # Expected log-likelihood — shape (B, P)
        sq_err = (y - mean) ** 2                 # (B, P)
        ell = -0.5 * (
            math.log(2 * math.pi)
            + noise.log()                        # (P,)  broadcasts over B
            + sq_err / noise
            + var  / noise
        )                                        # (B, P)

        # Scale and sum over batch and outputs
        scaled_ell = (n_total / B) * ell.sum()

        kl   = self.kl_divergence()
        return scaled_ell - kl


# ─────────────────────────────────────────────
# 3.  TRAINING LOOP
# ─────────────────────────────────────────────

def train_svgp(
    model: SVGP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,        # (N, P)
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 0.01,
    print_every: int = 100,
) -> list:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_total   = X_train.shape[0]
    losses    = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        idx  = torch.randperm(n_total)[:batch_size]
        X_b  = X_train[idx]
        y_b  = y_train[idx]

        optimizer.zero_grad()
        loss = -model.elbo(X_b, y_b, n_total)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # if epoch % print_every == 0:
        #     print(f"Epoch {epoch:4d}/{n_epochs}  "
        #           f"−ELBO = {loss.item():.3f}  "
        #           f"ls = {model.kernel.lengthscale.item():.3f}  "
        #           f"noise = [{', '.join(f'{v:.3f}' for v in model.noise.tolist())}]")

    return torch.mean(losses).detach()


# ─────────────────────────────────────────────
# 4.  DEMO — 1D INPUT, 6 OUTPUTS
# ─────────────────────────────────────────────

def demo():
    torch.manual_seed(42)
    P = 6     # number of outputs
    n = 1000  # training points
    m = 40    # inducing points

    # ── Synthetic data: 6 different functions of x ──────────────
    X_train = torch.rand(n, 1) * 10 - 5                # (n, 1)  U[−5, 5]
    x       = X_train.squeeze()

    # Each output is a different function + noise
    y_train = torch.stack([
        torch.sin(x),
        torch.cos(x),
        torch.sinc(x),
        0.5 * x,
        torch.sin(2 * x) * torch.exp(-0.2 * x.abs()),
        x ** 2 / 10,
    ], dim=1) + 0.1 * torch.randn(n, P)                # (n, P=6)

    output_names = ["sin(x)", "cos(x)", "sinc(x)", "0.5x",
                    "sin(2x)·exp(−0.2|x|)", "x²/10"]

    # ── Model ────────────────────────────────────────────────────
    Z_init = torch.linspace(-5, 5, m).unsqueeze(-1)    # (m, 1)
    kernel = RBFKernel(lengthscale=1.0, outputscale=1.0)
    model  = SVGP(kernel=kernel, inducing_pts=Z_init,
                             n_outputs=P, noise=0.1)

    # ── Train ────────────────────────────────────────────────────
    print(f"Training multi-output SVGP  (P={P} outputs, m={m} inducing pts)\n")
    losses = train_svgp(
        model, X_train, y_train,
        n_epochs=800, batch_size=256, lr=0.02, print_every=200
    )

    # ── Predict ──────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_test      = torch.linspace(-6, 6, 300).unsqueeze(-1)     # (300, 1)
        mu, var     = model.predict(X_test)                         # (300, 6), (300, 6)
        noise       = model.noise                                    # (6,)
        std         = (var + noise).sqrt()                           # (300, 6)

    print(f"\nPrediction shape — mean: {mu.shape}, var: {var.shape}")
    # Expected output:
    # Prediction shape — mean: torch.Size([300, 6]), var: torch.Size([300, 6])

    # ── Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Multi-Output SVGP (P={P})  —  PyTorch from scratch",
        fontsize=14, fontweight="bold"
    )
    axes = axes.flatten()

    x_tr = X_train.squeeze().numpy()
    x_te = X_test.squeeze().numpy()

    for p in range(P):
        ax = axes[p]
        ax.scatter(x_tr, y_train[:, p].numpy(),
                   s=4, alpha=0.25, color="steelblue", label="Data")
        ax.plot(x_te, mu[:, p].numpy(),
                color="crimson", lw=2, label="Mean")
        ax.fill_between(x_te,
                        (mu[:, p] - 2 * std[:, p]).numpy(),
                        (mu[:, p] + 2 * std[:, p]).numpy(),
                        alpha=0.2, color="crimson", label="±2 std")
        ax.vlines(model.Z.detach().squeeze().numpy(),
                  ax.get_ylim()[0], ax.get_ylim()[1],
                  colors="green", lw=0.6, alpha=0.5)
        ax.set_title(f"Output {p+1}: {output_names[p]}", fontsize=10)
        ax.set_xlabel("x")
        if p % 3 == 0:
            ax.set_ylabel("y")
        if p == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/svgp_multioutput_demo.png",
                dpi=150, bbox_inches="tight")
    print("Plot saved → svgp_multioutput_demo.png")
    plt.close()

    # ── Loss curve ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(losses, color="darkorange", lw=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("−ELBO")
    ax2.set_title("Training loss")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/svgp_multioutput_loss.png",
                dpi=150, bbox_inches="tight")
    print("Loss plot saved → svgp_multioutput_loss.png")
    plt.close()


if __name__ == "__main__":
    demo()