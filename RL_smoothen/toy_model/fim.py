"""FIM 相关计算：沿轨迹的 FIM 分析 + 可选的 2D 网格扫描。

改进：
- PCA 方向改为 d1 = normalize(θ_T - θ_0)，保证起止点在可视化平面上
- d2 = 轨迹在 d1 正交补中的最大方差方向
- 新增沿轨迹的 FIM 计算（始终运行，数值可靠）
- 2D 网格扫描作为可选功能保留
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from model import FlowModel
from env import reward


def compute_pca_directions(param_snapshots):
    """计算可视化用的 2D 方向和中心。

    d1 = normalize(θ_T - θ_0)：保证起止点严格在平面上。
    d2 = 轨迹在 d1 正交补中的最大方差方向。
    center = θ_0（起点为原点，终点在 α 轴上）。

    Returns:
        d1, d2: (D,) torch tensors, 单位向量
        center: (D,) torch tensor
        traj_alpha, traj_beta: (T,) numpy arrays, 轨迹在 2D 平面上的坐标
        explained_var: (2,) 方差解释比例
    """
    traj_params = torch.stack([fp for _, fp in param_snapshots])
    T, D = traj_params.shape

    # d1 = 起点到终点方向
    delta = traj_params[-1] - traj_params[0]
    d1 = delta / delta.norm()

    # 轨迹在 d1 上的投影
    center = traj_params[0].clone()
    centered = traj_params - center.unsqueeze(0)
    proj_alpha = (centered @ d1).numpy()  # (T,)

    # 减去 d1 分量，得到正交残差
    residuals = centered - torch.from_numpy(proj_alpha).unsqueeze(1) * d1.unsqueeze(0)

    # 对残差做 PCA 取第一主成分
    total_var = centered.var(dim=0).sum().item()
    d1_var = np.var(proj_alpha)

    residuals_np = residuals.numpy()
    res_var_total = np.var(residuals_np, axis=0).sum()

    if res_var_total > 1e-10:
        pca = PCA(n_components=1)
        pca.fit(residuals_np)
        d2 = torch.tensor(pca.components_[0], dtype=torch.float32)
        d2_var = pca.explained_variance_[0]
    else:
        # 轨迹几乎是 1D 的，取任意正交方向
        d2 = torch.randn(D)
        d2 = d2 - (d2 @ d1) * d1
        d2 = d2 / d2.norm()
        d2_var = 0.0

    # 确保 d2 严格正交于 d1（数值精度）
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / d2.norm()

    # 轨迹坐标
    traj_alpha = proj_alpha
    traj_beta = (centered @ d2).numpy()

    explained_var = np.array([d1_var / total_var, d2_var / total_var])
    print(f"  PCA directions: d1=start→end, d2=max orthogonal variance")
    print(f"  Explained variance: d1={explained_var[0]:.4f}, d2={explained_var[1]:.4f}, "
          f"total={explained_var.sum():.4f}")
    print(f"  θ_0 at (0, 0), θ_T at ({traj_alpha[-1]:.4f}, {traj_beta[-1]:.4f})")

    return d1, d2, center, traj_alpha, traj_beta, explained_var


def evaluate_J(model: FlowModel, flat_params: torch.Tensor, n_samples=500):
    """给定参数 θ（flat），计算 J(θ) = E[r(x)]。"""
    model.set_flat_params(flat_params)
    model.eval()
    with torch.no_grad():
        x, _ = model.sample(n_samples)
        return reward(x).mean().item()


def _compute_score_projections(model, flat_params, directions, n_samples=100):
    """计算 score 在多个方向上的投影，同时返回配对的 reward。

    Args:
        model: 生成模型
        flat_params: 参数 θ
        directions: list of (D,) tensors，投影方向
        n_samples: 采样数

    Returns:
        projections: list of lists，每个方向的投影值
        rewards: list，配对的 reward 值（与投影一一对应）
    """
    model.set_flat_params(flat_params)
    model.train()

    n_dirs = len(directions)
    projections = [[] for _ in range(n_dirs)]
    rewards = []

    for _ in range(n_samples):
        x, _ = model.sample(1)
        x = x.detach()

        r_val = reward(x).item()
        if not np.isfinite(r_val):
            continue

        log_prob = model.log_prob(x)

        if not torch.isfinite(log_prob):
            continue

        model.zero_grad()
        log_prob.backward()
        grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

        if not torch.isfinite(grad).all():
            continue

        rewards.append(r_val)
        for k, d in enumerate(directions):
            projections[k].append((grad @ d).item())

    return projections, rewards


def _trimmed_mean_square(vals, trim_ratio=0.1):
    """对投影值的平方做 trimmed mean（带 MAD 过滤）。"""
    if len(vals) == 0:
        return 1e-3
    vals_sq = np.array([v**2 for v in vals])

    median = np.median(vals_sq)
    mad = np.median(np.abs(vals_sq - median))

    if mad > 1e-10:
        mask = np.abs(vals_sq - median) <= 5 * mad
        vals_sq = vals_sq[mask]

    if len(vals_sq) == 0:
        return median if median > 1e-10 else 1e-3

    n = len(vals_sq)
    n_trim = int(n * trim_ratio)
    if n_trim > 0 and n > 2 * n_trim:
        vals_sorted = np.sort(vals_sq)
        return vals_sorted[n_trim:-n_trim].mean()
    else:
        return vals_sq.mean()


def compute_fim_diagonal(model, flat_params, d1, d2, n_samples=100, trim_ratio=0.1):
    """计算单点的 FIM 对角投影：F_kk = E[(∇_θ log π · d_k)²]。"""
    projs, _ = _compute_score_projections(model, flat_params, [d1, d2], n_samples)
    f11 = _trimmed_mean_square(projs[0], trim_ratio)
    f22 = _trimmed_mean_square(projs[1], trim_ratio)
    return f11, f22


# ======== 沿轨迹的 FIM 计算 ========

def compute_trajectory_fim(model, param_snapshots, d1, d2,
                           n_samples=100, trim_ratio=0.1):
    """沿 PPO 轨迹计算 FIM 投影、KL 弧长和 Theorem 1 配对统计量。

    在每个快照 θ_t 上计算：
    - F11(t), F22(t)：d1, d2 方向的 FIM 投影
    - s_KL(t→t+1)：到下一个快照的 KL 弧长
    - s_Euclid(t→t+1)：欧氏步长
    - C_v(t)：配对 Cauchy-Schwarz 统计量 = Cov(r, g·v)² / Var(g·v)
    - reward_var(t)：配对样本的 Var(r)

    Returns:
        dict with arrays indexed by trajectory step
    """
    traj_params = torch.stack([fp for _, fp in param_snapshots])
    T = len(traj_params)

    F11_traj = np.zeros(T)
    F22_traj = np.zeros(T)
    s_kl = np.zeros(T - 1)
    s_euclid = np.zeros(T - 1)
    C_v = np.zeros(T - 1)
    paired_reward_var = np.zeros(T)

    for t in range(T):
        theta_t = traj_params[t]

        directions = [d1, d2]
        if t < T - 1:
            delta = traj_params[t + 1] - theta_t
            s_euclid[t] = delta.norm().item()
            if s_euclid[t] > 1e-10:
                directions.append(delta)
            else:
                directions.append(d1)

        projs, r_paired = _compute_score_projections(
            model, theta_t, directions, n_samples)

        F11_traj[t] = _trimmed_mean_square(projs[0], trim_ratio)
        F22_traj[t] = _trimmed_mean_square(projs[1], trim_ratio)

        r_arr = np.array(r_paired)
        if len(r_arr) > 1:
            paired_reward_var[t] = np.var(r_arr, ddof=1)

        if t < T - 1:
            f_delta = _trimmed_mean_square(projs[2], trim_ratio)
            s_kl[t] = np.sqrt(0.5 * f_delta)

            gv = np.array(projs[2])
            if len(gv) > 1 and len(r_arr) == len(gv):
                E_r_gv = np.mean(r_arr * gv)
                E_gv2 = np.mean(gv ** 2)
                if E_gv2 > 1e-10:
                    C_v[t] = E_r_gv ** 2 / E_gv2

        if (t + 1) % 20 == 0 or t == T - 1:
            print(f"    Trajectory FIM: {t+1}/{T} done, "
                  f"F11={F11_traj[t]:.4f}, F22={F22_traj[t]:.4f}")

    cumulative_kl = np.concatenate([[0], np.cumsum(s_kl)])
    cumulative_euclid = np.concatenate([[0], np.cumsum(s_euclid)])
    rho = np.zeros(T - 1)
    mask = s_euclid > 1e-10
    rho[mask] = s_kl[mask] / s_euclid[mask]

    print(f"    F11 range: [{F11_traj.min():.4f}, {F11_traj.max():.4f}]")
    print(f"    F22 range: [{F22_traj.min():.4f}, {F22_traj.max():.4f}]")
    print(f"    Total KL arc length: {cumulative_kl[-1]:.4f}")
    print(f"    Total Euclidean arc length: {cumulative_euclid[-1]:.4f}")

    return {
        'F11': F11_traj,
        'F22': F22_traj,
        's_kl': s_kl,
        's_euclid': s_euclid,
        'rho': rho,
        'cumulative_kl': cumulative_kl,
        'cumulative_euclid': cumulative_euclid,
        'C_v': C_v,
        'paired_reward_var': paired_reward_var,
    }


# ======== 2D 网格扫描（可选，--scan 时使用） ========

def compute_fim_field(model, center, d1, d2, alphas, betas, n_samples=100,
                      trim_ratio=0.1):
    """在整个 2D 网格上计算 FIM 对角分量 F11, F22。"""
    na, nb = len(alphas), len(betas)
    F11 = np.zeros((nb, na))
    F22 = np.zeros((nb, na))

    total = na * nb
    done = 0
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            theta = center + alpha * d1 + beta * d2
            F11[j, i], F22[j, i] = compute_fim_diagonal(
                model, theta, d1, d2, n_samples, trim_ratio)
            done += 1
        if (i + 1) % 5 == 0:
            print(f"    FIM column {i+1}/{na} done ({done}/{total})")

    return F11, F22


def build_kl_coords_2d(alphas, betas, F11, F22):
    """从 2D FIM 场构建 KL 坐标（近似，假设 F12≈0）。"""
    na = len(alphas)
    nb = len(betas)
    ca = na // 2
    cb = nb // 2

    kl_alpha = np.zeros((nb, na))
    kl_beta = np.zeros((nb, na))

    dalpha = np.diff(alphas)
    dbeta = np.diff(betas)
    fim_floor = 1e-3

    for j in range(nb):
        for i in range(ca, na - 1):
            f_mid = max(0.5 * (F11[j, i] + F11[j, i + 1]), fim_floor)
            kl_alpha[j, i + 1] = kl_alpha[j, i] + \
                np.sqrt(0.5 * f_mid) * abs(dalpha[i])
        for i in range(ca, 0, -1):
            f_mid = max(0.5 * (F11[j, i] + F11[j, i - 1]), fim_floor)
            kl_alpha[j, i - 1] = kl_alpha[j, i] - \
                np.sqrt(0.5 * f_mid) * abs(dalpha[i - 1])

    for i in range(na):
        for j in range(cb, nb - 1):
            f_mid = max(0.5 * (F22[j, i] + F22[j + 1, i]), fim_floor)
            kl_beta[j + 1, i] = kl_beta[j, i] + \
                np.sqrt(0.5 * f_mid) * abs(dbeta[j])
        for j in range(cb, 0, -1):
            f_mid = max(0.5 * (F22[j, i] + F22[j - 1, i]), fim_floor)
            kl_beta[j - 1, i] = kl_beta[j, i] - \
                np.sqrt(0.5 * f_mid) * abs(dbeta[j - 1])

    return kl_alpha, kl_beta


def compute_theta_landscape(
    model: FlowModel,
    param_snapshots: list,
    d1, d2, center, traj_alpha, traj_beta,
    n_grid=41,
    grid_range=3.0,
    n_eval_samples=500,
    n_fim_samples=100,
    trim_ratio=0.1,
):
    """在 θ 空间扫描 J(θ)，同时计算完整的 2D FIM 场并构建 KL 坐标。

    使用外部传入的 d1, d2, center（由 compute_pca_directions 生成）。
    """
    # 确定欧氏网格范围
    alpha_min, alpha_max = traj_alpha.min(), traj_alpha.max()
    beta_min, beta_max = traj_beta.min(), traj_beta.max()
    alpha_margin = max((alpha_max - alpha_min) * (grid_range - 1) / 2, 0.01)
    beta_margin = max((beta_max - beta_min) * (grid_range - 1) / 2, 0.01)

    alphas = np.linspace(alpha_min - alpha_margin, alpha_max + alpha_margin, n_grid)
    betas = np.linspace(beta_min - beta_margin, beta_max + beta_margin, n_grid)

    # === 扫描 J(θ) 在欧氏网格上 ===
    J_grid = np.zeros((n_grid, n_grid))
    print(f"  Scanning {n_grid}x{n_grid} = {n_grid**2} grid points for J(theta)...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            theta = center + alpha * d1 + beta * d2
            J_grid[j, i] = evaluate_J(model, theta, n_eval_samples)
        if (i + 1) % 10 == 0:
            print(f"    J column {i+1}/{n_grid} done")

    # === 计算完整 2D FIM 场 ===
    print(f"  Computing full 2D FIM field at {n_grid}x{n_grid} points "
          f"({n_fim_samples} samples/point, trimmed mean with trim={trim_ratio})...")
    F11, F22 = compute_fim_field(
        model, center, d1, d2, alphas, betas, n_fim_samples, trim_ratio)
    print(f"    F11 range: [{F11.min():.6f}, {F11.max():.6f}]")
    print(f"    F22 range: [{F22.min():.6f}, {F22.max():.6f}]")
    print(f"    F11 ratio (max/min): {F11.max() / max(F11.min(), 1e-10):.2f}")
    print(f"    F22 ratio (max/min): {F22.max() / max(F22.min(), 1e-10):.2f}")

    # === 构建 2D KL 坐标 ===
    kl_alpha_2d, kl_beta_2d = build_kl_coords_2d(alphas, betas, F11, F22)
    print(f"    KL-alpha range: [{kl_alpha_2d.min():.4f}, {kl_alpha_2d.max():.4f}]")
    print(f"    KL-beta  range: [{kl_beta_2d.min():.4f}, {kl_beta_2d.max():.4f}]")

    # === 均匀 KL 网格 ===
    F11_avg = F11.mean(axis=0)
    F22_avg = F22.mean(axis=1)

    # 从 alpha=0 处开始积分（起点是原点）
    dalpha = np.diff(alphas)
    dbeta = np.diff(betas)
    fim_floor = 1e-3

    # 找到最接近 0 的索引作为积分起点
    ca = np.argmin(np.abs(alphas))
    cb = np.argmin(np.abs(betas))

    kl_alpha_1d = np.zeros(n_grid)
    for i in range(ca, n_grid - 1):
        f_mid = max(0.5 * (F11_avg[i] + F11_avg[i + 1]), fim_floor)
        kl_alpha_1d[i + 1] = kl_alpha_1d[i] + np.sqrt(0.5 * f_mid) * abs(dalpha[i])
    for i in range(ca, 0, -1):
        f_mid = max(0.5 * (F11_avg[i] + F11_avg[i - 1]), fim_floor)
        kl_alpha_1d[i - 1] = kl_alpha_1d[i] - np.sqrt(0.5 * f_mid) * abs(dalpha[i - 1])

    kl_beta_1d = np.zeros(n_grid)
    for j in range(cb, n_grid - 1):
        f_mid = max(0.5 * (F22_avg[j] + F22_avg[j + 1]), fim_floor)
        kl_beta_1d[j + 1] = kl_beta_1d[j] + np.sqrt(0.5 * f_mid) * abs(dbeta[j])
    for j in range(cb, 0, -1):
        f_mid = max(0.5 * (F22_avg[j] + F22_avg[j - 1]), fim_floor)
        kl_beta_1d[j - 1] = kl_beta_1d[j] - np.sqrt(0.5 * f_mid) * abs(dbeta[j - 1])

    traj_alpha_kl = np.interp(traj_alpha, alphas, kl_alpha_1d)
    traj_beta_kl = np.interp(traj_beta, betas, kl_beta_1d)

    kl_alphas_uniform = np.linspace(kl_alpha_1d.min(), kl_alpha_1d.max(), n_grid)
    kl_betas_uniform = np.linspace(kl_beta_1d.min(), kl_beta_1d.max(), n_grid)

    alpha_from_kl = np.interp(kl_alphas_uniform, kl_alpha_1d, alphas)
    beta_from_kl = np.interp(kl_betas_uniform, kl_beta_1d, betas)

    J_kl_grid = np.zeros((n_grid, n_grid))
    print(f"  Scanning J(theta) on uniform KL grid ({n_grid}x{n_grid})...")
    for i, kl_a in enumerate(kl_alphas_uniform):
        a = alpha_from_kl[i]
        for j, kl_b in enumerate(kl_betas_uniform):
            b = beta_from_kl[j]
            theta = center + a * d1 + b * d2
            J_kl_grid[j, i] = evaluate_J(model, theta, n_eval_samples)
        if (i + 1) % 10 == 0:
            print(f"    KL-J column {i+1}/{n_grid} done")

    return {
        'alphas': alphas,
        'betas': betas,
        'J_grid': J_grid,
        'traj_alpha': traj_alpha,
        'traj_beta': traj_beta,
        'explained_var': np.array([0.0, 0.0]),  # placeholder, use from compute_pca_directions
        'kl_alpha_2d': kl_alpha_2d,
        'kl_beta_2d': kl_beta_2d,
        'F11_field': F11,
        'F22_field': F22,
        'kl_alphas_uniform': kl_alphas_uniform,
        'kl_betas_uniform': kl_betas_uniform,
        'J_kl_grid': J_kl_grid,
        'traj_alpha_kl': traj_alpha_kl,
        'traj_beta_kl': traj_beta_kl,
    }
