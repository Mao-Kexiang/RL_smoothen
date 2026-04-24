"""CMA-ES 直接优化：在 x 空间中直接搜索 r(x) 的最优解。"""

import numpy as np
from env import reward_np


def cmaes_optimize(
    n_evals=50000,
    sigma0=3.0,
    seed=42,
):
    """用 CMA-ES + random restarts 直接在 x 空间优化 Rastrigin 函数。

    Args:
        n_evals: 最大总函数评估次数
        n_restarts: 重启次数
        sigma0: 初始步长
        seed: 随机种子

    Returns:
        history: dict
    """
    import cma

    rng = np.random.RandomState(seed)

    history = {
        'trajectory': [],
        'best_rewards': [],
        'mean_rewards': [],
        'all_samples': [],
        'n_evals_list': [],
    }

    total_evals = 0
    global_best_x = None
    global_best_r = -np.inf

    # 插入初始点 (eval=0)
    x0_init = rng.uniform(-5, 5, size=2)
    r_init = reward_np(x0_init)
    global_best_x = x0_init.copy()
    global_best_r = r_init
    history['trajectory'].append(x0_init.copy())
    history['best_rewards'].append(r_init)
    history['mean_rewards'].append(r_init)
    history['all_samples'].append(x0_init.reshape(1, -1))
    history['n_evals_list'].append(1)
    total_evals = 1

    restart = 0
    es = None

    while total_evals < n_evals:
        if es is None or es.stop():
            x0 = rng.uniform(-5, 5, size=2)
            es = cma.CMAEvolutionStrategy(
                x0.tolist(),
                sigma0,
                {'maxfevals': n_evals - total_evals, 'seed': seed + restart,
                 'verbose': -9, 'bounds': [[-5, -5], [5, 5]]},
            )
            restart += 1

        solutions = es.ask()
        fitnesses = [-reward_np(np.array(s)) for s in solutions]
        es.tell(solutions, fitnesses)

        total_evals += len(solutions)
        best_r_this = -min(fitnesses)

        if best_r_this > global_best_r:
            global_best_r = best_r_this
            global_best_x = es.result.xbest.copy()

        history['trajectory'].append(global_best_x.copy())
        history['best_rewards'].append(global_best_r)
        history['mean_rewards'].append(-np.mean(fitnesses))
        history['all_samples'].append(np.array(solutions))
        history['n_evals_list'].append(total_evals)

        if total_evals % 5000 < 10:
            print(f"  CMA-ES evals={total_evals}/{n_evals}: "
                  f"global_best_r={global_best_r:.4f}, restarts={restart}")

    history['final_best'] = global_best_x
    history['final_reward'] = global_best_r
    print(f"  CMA-ES done: best_r={global_best_r:.4f}, total_evals={total_evals}")
    return history
