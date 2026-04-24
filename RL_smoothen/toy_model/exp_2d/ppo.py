"""PPO 训练（流模型专用）：含发散恢复和 NaN 降 lr 机制。"""

import torch
import copy
import math
import numpy as np

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env import reward


def ppo_train(
    model,
    n_iters=1000,
    batch_size=256,
    ppo_epochs=2,
    clip_eps=0.2,
    lr=3e-4,
    kl_coeff=0.5,
    save_every=1,
    diverge_threshold=5.0,
    device='cpu',
):
    base_model = copy.deepcopy(model)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    min_lr = lr * 0.05

    best_mean_r = float('-inf')
    best_params = model.get_flat_params().clone().cpu()
    nan_count = 0
    diverge_count = 0

    history = {
        'mean_reward': [],
        'best_reward': [],
        'reward_var': [],
        'reward_rmax_eff': [],
        'kl_div': [],
        'param_snapshots': [],
        'sample_snapshots': [],
    }

    history['param_snapshots'].append((0, model.get_flat_params().clone().cpu()))
    with torch.no_grad():
        x_init, _ = model.sample(batch_size)
        init_r = reward(x_init)
        history['mean_reward'].append(init_r.mean().item())
        history['best_reward'].append(init_r.max().item())
        x_eval0, _ = model.sample(2048)
        r_eval0 = reward(x_eval0)
        history['reward_var'].append(r_eval0.var().item())
        history['reward_rmax_eff'].append(r_eval0.abs().max().item())
        history['kl_div'].append(0.0)
        history['sample_snapshots'].append((0, x_init[:500].cpu().numpy()))

    for it in range(n_iters):
        model.eval()
        with torch.no_grad():
            x, old_log_prob = model.sample(batch_size)
            r = reward(x)

            valid = (torch.isfinite(x).all(dim=-1)
                     & torch.isfinite(old_log_prob) & torch.isfinite(r))
            if valid.sum() < 32:
                nan_count += 1
                model.set_flat_params(best_params.clone())
                for pg in optimizer.param_groups:
                    pg['lr'] = max(pg['lr'] * 0.5, min_lr)
                history['mean_reward'].append(history['mean_reward'][-1])
                history['best_reward'].append(history['best_reward'][-1])
                history['reward_var'].append(history['reward_var'][-1])
                history['reward_rmax_eff'].append(history['reward_rmax_eff'][-1])
                history['kl_div'].append(history['kl_div'][-1])
                continue

            x = x[valid]
            old_log_prob = old_log_prob[valid]
            r = r[valid]
            r_normalized = (r - r.mean()) / (r.std() + 1e-8)

        model.train()
        step_ok = True
        for _ in range(ppo_epochs):
            new_log_prob = model.log_prob(x)

            bad = ~torch.isfinite(new_log_prob)
            if bad.any():
                new_log_prob = torch.where(bad, old_log_prob, new_log_prob)

            log_ratio = (new_log_prob - old_log_prob).clamp(-5, 5)
            ratio = log_ratio.exp()

            surr1 = ratio * r_normalized
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * r_normalized
            policy_loss = -torch.min(surr1, surr2).mean()

            with torch.no_grad():
                base_log_prob = base_model.log_prob(x)
                bad_base = ~torch.isfinite(base_log_prob)
                if bad_base.any():
                    base_log_prob = torch.where(
                        bad_base, new_log_prob.detach(), base_log_prob)
            kl = (new_log_prob - base_log_prob).clamp(-20, 20).mean()

            loss = policy_loss + kl_coeff * kl

            if not torch.isfinite(loss):
                model.set_flat_params(best_params.clone())
                nan_count += 1
                step_ok = False
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if abs(kl.item()) > 50:
                break

        with torch.no_grad():
            mean_r = r.mean().item()
            best_r = r.max().item()
            if step_ok:
                base_lp = base_model.log_prob(x)
                new_lp = model.log_prob(x)
                kl_val = (new_lp - base_lp).clamp(-20, 20).mean().item()
                if math.isnan(kl_val):
                    kl_val = history['kl_div'][-1] if history['kl_div'] else 0.0
            else:
                kl_val = history['kl_div'][-1] if history['kl_div'] else 0.0

        if (not math.isnan(mean_r) and best_mean_r > float('-inf')
                and mean_r < best_mean_r - diverge_threshold):
            model.set_flat_params(best_params.clone())
            diverge_count += 1
            mean_r = best_mean_r

        history['mean_reward'].append(mean_r)
        history['best_reward'].append(best_r)

        model.eval()
        with torch.no_grad():
            x_eval, _ = model.sample(2048)
            r_eval = reward(x_eval)
            valid_eval = torch.isfinite(r_eval)
            if valid_eval.sum() > 0:
                r_eval = r_eval[valid_eval]
            history['reward_var'].append(r_eval.var().item())
            history['reward_rmax_eff'].append(r_eval.abs().max().item())

        history['kl_div'].append(kl_val)

        if not math.isnan(mean_r) and mean_r > best_mean_r:
            best_mean_r = mean_r
            best_params = model.get_flat_params().clone().cpu()

        if (it + 1) % save_every == 0:
            history['param_snapshots'].append(
                (it + 1, model.get_flat_params().clone().cpu()))
            model.eval()
            with torch.no_grad():
                samples, _ = model.sample(500)
            samp_np = samples.cpu().numpy()
            samp_np = np.nan_to_num(samp_np, nan=0.0)
            history['sample_snapshots'].append((it + 1, samp_np))

        if (it + 1) % 20 == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  PPO iter {it+1}/{n_iters}: "
                  f"mean_r={mean_r:.2f}, best_r={best_r:.2f}, "
                  f"kl={kl_val:.4f}, lr={cur_lr:.2e}"
                  + (f" [nan:{nan_count}]" if nan_count > 0 else "")
                  + (f" [div:{diverge_count}]" if diverge_count > 0 else ""))

    if best_params is not None:
        model.set_flat_params(best_params)
        print(f"  Restored best model (mean_r={best_mean_r:.4f})")

    return history
