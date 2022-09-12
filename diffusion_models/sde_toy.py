import time
import matplotlib.pyplot as plt

import torch
import torchsde


class DDPM_Ex(torch.nn.Module):

    def __init__(self, beta_min=0.0001, beta_max=0.02, T=1000, T_rev=200) -> None:
        super().__init__()

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.T_rev = T_rev

        self.Beta = torch.linspace(beta_min, beta_max, T)
        self.Alpha = 1 - self.Beta
        self.Alpha_bar = torch.cumprod(self.Alpha, dim=0)

    def eps_t(self, x_t, t):

        return x_t
    
    def diffusion(self, x_0):

        with torch.no_grad():
            z = torch.normal(0, 1, size=x_0.shape).to(x_0.device)
            x_t = torch.sqrt(self.Alpha_bar[self.T_rev-1]).to(x_0.device) * x_0 + torch.sqrt(1-self.Alpha_bar[self.T_rev-1]).to(x_0.device) * z
        
        return x_t

    def reverse(self, x_t):

        with torch.no_grad():
            for t in range(self.T_rev-1, -1, -1):
                eps_t = self.eps_t(x_t, t)
                mu_t = (x_t - (1 - self.Alpha[t]) / torch.sqrt(1 - self.Alpha_bar[t]) * eps_t) / torch.sqrt(self.Alpha[t])
                if t > 0:
                    sigma_t = torch.sqrt(self.Beta[t] * (1 - self.Alpha_bar[t-1]) / (1 - self.Alpha_bar[t]))
                    x_t = mu_t.to(x_t.device) + sigma_t.to(x_t.device) * torch.normal(0, 1, size=x_t.shape).to(x_t.device)
                else: 
                    x_t = mu_t.to(x_t.device)
        return x_t


class RevVPSDE_Ex(torch.nn.Module):
    def __init__(self, mu_0=1., sigma2_0=0.01, beta_min=0.1, beta_max=20):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        super().__init__()

        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))

        self.mu_t = lambda t: mu_0 * torch.sqrt(self.alphas_cumprod_cont(t))
        self.sigma2_t = lambda t: 1 - (1 - sigma2_0) * self.alphas_cumprod_cont(t)

        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def rvpsde_fn(self, t, x, return_type='drift'):
        """Create the drift and diffusion functions for the reverse SDE
        """
        drift, diffusion = self.vpsde_fn(t, x)

        if return_type == 'drift':
            # score = - (x - self.mu_t(t)[:, None]) / self.sigma2_t(t)[:, None]
            score = self.sqrt_1m_alphas_cumprod_neg_recip_cont(t[0]) * x

            drift = drift - diffusion[:, None] ** 2 * score
            return drift

        else:
            return diffusion

    def f(self, t, x):
        """Create the drift function -f(x, 1-t) (by t' = 1 - t)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        drift = self.rvpsde_fn(1 - t, x, return_type='drift')
        assert drift.shape == x.shape, f'drift.shape: {drift.shape}, x.shape: {x.shape}'
        return -drift

    def g(self, t, x):
        """Create the diffusion function g(1-t) (by t' = 1 - t)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)


def main():
    grads = []
    diffs_list = []
    dts = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    sigmas2_0 = [0.01, 0.05, 0.1, 0.5, 1.0]
    markers = ['*', '.', '<', '+', '^']
    for sigma2_0 in sigmas2_0:
        batch_size = 4
        mu_0 = 0.
        sigma2_0 = sigma2_0
        rev_vpsde = RevVPSDE_Ex(mu_0=mu_0, sigma2_0=sigma2_0)
        ddpm = DDPM_Ex(T_rev=100)

        t_start = 0.1
        t0, t1 = 1 - t_start, 1
        t_size = 2
        ts = torch.linspace(t0, t1, t_size)

        diffs = []
        for dt in dts:
            args_dict = {'method': 'euler', 'dt': dt}

            n_iter = 1
            for i in range(n_iter):
                print(f'--------------------------- iter: {i} -----------------------')
                batch_size = batch_size - i

                t = torch.ones((batch_size, 1)) * t_start
                x = rev_vpsde.mu_t(t) + torch.randn((batch_size, 1)) * torch.sqrt(rev_vpsde.sigma2_t(t))
                x_start = x.clone().detach().requires_grad_(True)
                print(f'x_start: {x_start}')

                bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, 1), entropy=124)
                xs = torchsde.sdeint_adjoint(rev_vpsde, x_start, ts, bm=bm, **args_dict)

                x_ddpm = ddpm.reverse(x_start)

                x0_re = xs[-1]
                start_time = time.time()
                grad_adj, = torch.autograd.grad(torch.sum(x0_re), x_start)
                print(f'x0_re [adj]: {x0_re}')
                print(f'grad [adj]: {grad_adj}, with elapsed time: {time.time() - start_time}')

                # sdeint
                xs_ = torchsde.sdeint(rev_vpsde, x_start, ts, bm=bm, **args_dict)
                x0_re_ = xs_[-1]
                start_time = time.time()
                grad, = torch.autograd.grad(torch.sum(x0_re_), x_start)
                print(f'x0_re [direct]: {x0_re_}')
                print(f'grad [direct]: {grad}, with elapsed time: {time.time() - start_time}')

                # grad (analytic)
                alpha_t = rev_vpsde.alphas_cumprod_cont(torch.tensor(t_start))
                grad_ana = sigma2_0 * torch.sqrt(alpha_t) / (1 - alpha_t + sigma2_0 * alpha_t)
                print(f'alpha_t: {alpha_t.item()}')
                print(f'grad [analytic]: {grad_ana.item()}')

                print()
                print(f'args_dict: {args_dict}')
                print(f'diff between analytic and adj: {np.abs(grad_ana.item() - grad_adj[0].item())}')
                print(f'diff between analytic and direct: {np.abs(grad_ana.item() - grad[0].item())}')
                diffs.append(np.abs(grad_ana.item() - grad_adj[0].item()) / np.abs(grad_ana.item()))

        diffs_list.append(diffs)
        grads.append(grad_ana.item())

    # grads = [ (0.0840),  (0.3187),  (0.4897),  (0.8578),  (0.9467)]
    # diffs_list = [[0.9405032616313309, 0.28064550513318637, 0.028167590236131627, 0.002820767281688938, 0.00028341577013668605], [1.5259425298964984, 0.013862642562439449, 0.001529425186747912, 0.0001539244180609638, 1.4027134088180177e-05], [0.8482535731186445, 0.017624920305721707, 0.0015365817659491266, 0.00015094362590326523, 1.3390160362386431e-05], [0.14834377242756056, 0.011817430832082995, 0.0011527522907729228, 0.00011403137676522795, 1.0979255045037183e-05], [0.050995142336418074, 0.004821021451637803, 0.00047855125831090106, 4.778587094566161e-05, 4.847841979994656e-06]]

    print(f'diffs_list: {diffs_list}')
    print(f'grads_ana: {grads}')
    plt.grid(visible=True, alpha=0.5, ls='--')
    for i, diffs in enumerate(diffs_list):
        plt.plot(dts, diffs, marker=markers[i])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('step size', fontsize=11)
    plt.ylabel('Numerical error of gradients', fontsize=11)
    plt.legend([f'$\sigma_0^2$={sigma2_0}' for sigma2_0 in sigmas2_0])
    plt.savefig('impact_step_size.pdf')
    plt.show()


if __name__ == '__main__':
    # set random seed
    seed = 124
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    main()