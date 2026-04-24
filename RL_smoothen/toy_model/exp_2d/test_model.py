"""流模型 (FlowModel) 单元测试：验证可逆性、log-det 正确性、log_prob 自洽性、归一化。

包含两组测试：
- 随机初始化参数（接近恒等变换）
- Boltzmann(beta=1.0) 预训练后的参数（多峰、非平凡变换）
"""

import os
import sys
import pytest
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import FlowModel, AffineCoupling, SplineCoupling
from boltzmann import pretrain_boltzmann


SEED = 42
BATCH = 64
ATOL_INV = 1e-5
ATOL_LOGDET = 1e-4
ATOL_LOGPROB = 1e-5


# ======== Fixtures: 随机初始化 ========

@pytest.fixture
def spline_model():
    torch.manual_seed(SEED)
    return FlowModel(n_layers=8, hidden_dim=64, coupling='spline').eval()


@pytest.fixture
def affine_model():
    torch.manual_seed(SEED)
    return FlowModel(n_layers=8, hidden_dim=64, coupling='affine').eval()


@pytest.fixture
def spline_layer():
    torch.manual_seed(SEED)
    return SplineCoupling(fix_dim=0, hidden_dim=64).eval()


@pytest.fixture
def affine_layer():
    torch.manual_seed(SEED)
    return AffineCoupling(fix_dim=0, hidden_dim=64).eval()


# ======== Fixtures: Boltzmann(beta=1.0) 预训练 ========

@pytest.fixture(scope='module')
def trained_spline_model():
    torch.manual_seed(SEED)
    model = FlowModel(n_layers=8, hidden_dim=64, coupling='spline')
    pretrain_boltzmann(model, n_epochs=300, batch_size=512, lr=1e-3, beta=1.0)
    return model.eval()


@pytest.fixture(scope='module')
def trained_affine_model():
    torch.manual_seed(SEED)
    model = FlowModel(n_layers=8, hidden_dim=64, coupling='affine')
    pretrain_boltzmann(model, n_epochs=300, batch_size=512, lr=1e-3, beta=1.0)
    return model.eval()


def _random_input(n=BATCH, bound=4.0):
    torch.manual_seed(SEED + 1)
    return torch.rand(n, 2) * 2 * bound - bound


def _random_z(n=BATCH):
    torch.manual_seed(SEED + 2)
    return torch.randn(n, 2)


def _autograd_log_abs_det(func, z):
    """用 autograd 逐样本计算 log|det J|。"""
    log_dets = []
    for i in range(z.shape[0]):
        zi = z[i:i+1].detach().requires_grad_(True)

        def f(inp):
            out, _ = func(inp)
            return out.squeeze(0)

        J = torch.autograd.functional.jacobian(f, zi, create_graph=False)
        # J shape: (2, 1, 2) → squeeze to (2, 2)
        J = J.squeeze(1)
        log_dets.append(torch.log(torch.abs(torch.det(J))))
    return torch.stack(log_dets)


# ======== 1. Invertibility Tests ========

class TestInvertibility:

    def test_spline_coupling_invertibility(self, spline_layer):
        x = _random_input()
        y, _ = spline_layer(x)
        x_rec, _ = spline_layer.inverse(y)
        assert torch.allclose(x, x_rec, atol=ATOL_INV), \
            f"SplineCoupling forward→inverse max err: {(x - x_rec).abs().max():.2e}"

    def test_affine_coupling_invertibility(self, affine_layer):
        x = _random_input()
        y, _ = affine_layer(x)
        x_rec, _ = affine_layer.inverse(y)
        assert torch.allclose(x, x_rec, atol=ATOL_INV), \
            f"AffineCoupling forward→inverse max err: {(x - x_rec).abs().max():.2e}"

    def test_flow_model_spline_invertibility(self, spline_model):
        z = _random_z()
        x, _ = spline_model.forward(z)
        z_rec, _ = spline_model.inverse(x)
        assert torch.allclose(z, z_rec, atol=ATOL_INV), \
            f"Spline FlowModel z→x→z max err: {(z - z_rec).abs().max():.2e}"

    def test_flow_model_affine_invertibility(self, affine_model):
        z = _random_z()
        x, _ = affine_model.forward(z)
        z_rec, _ = affine_model.inverse(x)
        assert torch.allclose(z, z_rec, atol=ATOL_INV), \
            f"Affine FlowModel z→x→z max err: {(z - z_rec).abs().max():.2e}"

    def test_inverse_then_forward(self, spline_model):
        x = _random_input()
        z, _ = spline_model.inverse(x)
        x_rec, _ = spline_model.forward(z)
        assert torch.allclose(x, x_rec, atol=ATOL_INV), \
            f"Spline FlowModel x→z→x max err: {(x - x_rec).abs().max():.2e}"


# ======== 2. Log-det Correctness Tests ========

class TestLogDet:

    def test_spline_coupling_logdet(self, spline_layer):
        x = _random_input(n=16)
        _, logdet_model = spline_layer(x)
        logdet_auto = _autograd_log_abs_det(spline_layer, x)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"SplineCoupling logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"

    def test_affine_coupling_logdet(self, affine_layer):
        x = _random_input(n=16)
        _, logdet_model = affine_layer(x)
        logdet_auto = _autograd_log_abs_det(affine_layer, x)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"AffineCoupling logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"

    def test_flow_model_forward_logdet(self, spline_model):
        z = _random_z(n=16)
        _, logdet_model = spline_model.forward(z)
        logdet_auto = _autograd_log_abs_det(spline_model.forward, z)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"Spline FlowModel forward logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"

    def test_flow_model_inverse_logdet(self, spline_model):
        x = _random_input(n=16)
        _, logdet_model = spline_model.inverse(x)
        logdet_auto = _autograd_log_abs_det(spline_model.inverse, x)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"Spline FlowModel inverse logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"

    def test_affine_flow_model_forward_logdet(self, affine_model):
        z = _random_z(n=16)
        _, logdet_model = affine_model.forward(z)
        logdet_auto = _autograd_log_abs_det(affine_model.forward, z)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"Affine FlowModel forward logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"


# ======== 3. Log-prob Consistency Tests ========

class TestLogProbConsistency:

    def test_sample_logprob_matches_logprob(self, spline_model):
        torch.manual_seed(SEED + 10)
        x, logprob_sample = spline_model.sample(BATCH)
        logprob_eval = spline_model.log_prob(x)
        assert torch.allclose(logprob_sample, logprob_eval, atol=ATOL_LOGPROB), \
            f"sample() vs log_prob() max err: {(logprob_sample - logprob_eval).abs().max():.2e}"

    def test_sample_logprob_matches_logprob_affine(self, affine_model):
        torch.manual_seed(SEED + 10)
        x, logprob_sample = affine_model.sample(BATCH)
        logprob_eval = affine_model.log_prob(x)
        assert torch.allclose(logprob_sample, logprob_eval, atol=ATOL_LOGPROB), \
            f"sample() vs log_prob() max err (affine): {(logprob_sample - logprob_eval).abs().max():.2e}"

    def test_logprob_decomposition(self, spline_model):
        x = _random_input()
        z, logdet_inv = spline_model.inverse(x)
        log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)
        logprob_manual = log_pz + logdet_inv
        logprob_model = spline_model.log_prob(x)
        assert torch.allclose(logprob_manual, logprob_model, atol=ATOL_LOGPROB), \
            f"log_prob decomposition max err: {(logprob_manual - logprob_model).abs().max():.2e}"


# ======== 4. Normalization Test ========

class TestNormalization:

    def test_density_integrates_to_one(self, spline_model):
        n_grid = 400
        bound = 6.0
        xs = torch.linspace(-bound, bound, n_grid)
        dx = xs[1] - xs[0]
        grid_x, grid_y = torch.meshgrid(xs, xs, indexing='ij')
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        with torch.no_grad():
            log_probs = []
            chunk = 4096
            for i in range(0, points.shape[0], chunk):
                log_probs.append(spline_model.log_prob(points[i:i+chunk]))
            log_probs = torch.cat(log_probs)

        integral = (log_probs.exp() * dx * dx).sum().item()
        assert abs(integral - 1.0) < 0.05, \
            f"Density integral = {integral:.4f}, expected ~1.0"


# ======== 5. Forward/Inverse Log-det Symmetry ========

class TestLogDetSymmetry:

    def test_forward_inverse_logdet_cancel(self, spline_model):
        z = _random_z()
        x, logdet_fwd = spline_model.forward(z)
        _, logdet_inv = spline_model.inverse(x)
        total = logdet_fwd + logdet_inv
        assert torch.allclose(total, torch.zeros_like(total), atol=ATOL_INV), \
            f"logdet_fwd + logdet_inv max err: {total.abs().max():.2e}"

    def test_forward_inverse_logdet_cancel_affine(self, affine_model):
        z = _random_z()
        x, logdet_fwd = affine_model.forward(z)
        _, logdet_inv = affine_model.inverse(x)
        total = logdet_fwd + logdet_inv
        assert torch.allclose(total, torch.zeros_like(total), atol=ATOL_INV), \
            f"logdet_fwd + logdet_inv max err (affine): {total.abs().max():.2e}"


# ================================================================
# 训练后模型测试 (Boltzmann beta=1.0 pretrained, 非平凡变换)
# ================================================================

class TestTrainedInvertibility:

    def test_trained_spline_invertibility(self, trained_spline_model):
        z = _random_z()
        x, _ = trained_spline_model.forward(z)
        z_rec, _ = trained_spline_model.inverse(x)
        assert torch.allclose(z, z_rec, atol=ATOL_INV), \
            f"Trained spline z→x→z max err: {(z - z_rec).abs().max():.2e}"

    def test_trained_affine_invertibility(self, trained_affine_model):
        z = _random_z()
        x, _ = trained_affine_model.forward(z)
        z_rec, _ = trained_affine_model.inverse(x)
        assert torch.allclose(z, z_rec, atol=ATOL_INV), \
            f"Trained affine z→x→z max err: {(z - z_rec).abs().max():.2e}"

    def test_trained_inverse_then_forward(self, trained_spline_model):
        x = _random_input()
        z, _ = trained_spline_model.inverse(x)
        x_rec, _ = trained_spline_model.forward(z)
        assert torch.allclose(x, x_rec, atol=ATOL_INV), \
            f"Trained spline x→z→x max err: {(x - x_rec).abs().max():.2e}"


class TestTrainedLogDet:

    def test_trained_spline_forward_logdet(self, trained_spline_model):
        z = _random_z(n=16)
        _, logdet_model = trained_spline_model.forward(z)
        logdet_auto = _autograd_log_abs_det(trained_spline_model.forward, z)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"Trained spline forward logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"

    def test_trained_spline_inverse_logdet(self, trained_spline_model):
        x = _random_input(n=16)
        _, logdet_model = trained_spline_model.inverse(x)
        logdet_auto = _autograd_log_abs_det(trained_spline_model.inverse, x)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"Trained spline inverse logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"

    def test_trained_affine_forward_logdet(self, trained_affine_model):
        z = _random_z(n=16)
        _, logdet_model = trained_affine_model.forward(z)
        logdet_auto = _autograd_log_abs_det(trained_affine_model.forward, z)
        assert torch.allclose(logdet_model, logdet_auto, atol=ATOL_LOGDET), \
            f"Trained affine forward logdet max err: {(logdet_model - logdet_auto).abs().max():.2e}"


class TestTrainedLogProbConsistency:

    def test_trained_sample_logprob_matches(self, trained_spline_model):
        torch.manual_seed(SEED + 20)
        x, logprob_sample = trained_spline_model.sample(BATCH)
        logprob_eval = trained_spline_model.log_prob(x)
        # 训练后参数大，forward/inverse 两条路径的 float32 舍入累积到 ~1e-5，放宽到 1e-4
        assert torch.allclose(logprob_sample, logprob_eval, atol=1e-4), \
            f"Trained sample vs log_prob max err: {(logprob_sample - logprob_eval).abs().max():.2e}"

    def test_trained_logprob_decomposition(self, trained_spline_model):
        x = _random_input()
        z, logdet_inv = trained_spline_model.inverse(x)
        log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)
        logprob_manual = log_pz + logdet_inv
        logprob_model = trained_spline_model.log_prob(x)
        assert torch.allclose(logprob_manual, logprob_model, atol=ATOL_LOGPROB), \
            f"Trained logprob decomposition max err: {(logprob_manual - logprob_model).abs().max():.2e}"


class TestTrainedNormalization:

    def test_trained_density_integrates_to_one(self, trained_spline_model):
        n_grid = 400
        bound = 6.0
        xs = torch.linspace(-bound, bound, n_grid)
        dx = xs[1] - xs[0]
        grid_x, grid_y = torch.meshgrid(xs, xs, indexing='ij')
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        with torch.no_grad():
            log_probs = []
            chunk = 4096
            for i in range(0, points.shape[0], chunk):
                log_probs.append(trained_spline_model.log_prob(points[i:i+chunk]))
            log_probs = torch.cat(log_probs)

        integral = (log_probs.exp() * dx * dx).sum().item()
        assert abs(integral - 1.0) < 0.05, \
            f"Trained density integral = {integral:.4f}, expected ~1.0"


class TestTrainedLogDetSymmetry:

    def test_trained_logdet_cancel(self, trained_spline_model):
        z = _random_z()
        x, logdet_fwd = trained_spline_model.forward(z)
        _, logdet_inv = trained_spline_model.inverse(x)
        total = logdet_fwd + logdet_inv
        # 训练后参数较大，8 层样条的 float32 舍入误差会累积，放宽到 1e-4
        assert torch.allclose(total, torch.zeros_like(total), atol=1e-4), \
            f"Trained logdet_fwd + logdet_inv max err: {total.abs().max():.2e}"


# ================================================================
# Float64 测试：用双精度证实 float32 误差纯属舍入，非代码 bug
# ================================================================

ATOL_F64 = 1e-10


@pytest.fixture(scope='module')
def trained_spline_f64():
    torch.manual_seed(SEED)
    model = FlowModel(n_layers=8, hidden_dim=64, coupling='spline')
    pretrain_boltzmann(model, n_epochs=300, batch_size=512, lr=1e-3, beta=1.0)
    return model.double().eval()


def _random_z_f64(n=BATCH):
    torch.manual_seed(SEED + 2)
    return torch.randn(n, 2, dtype=torch.float64)


def _random_input_f64(n=BATCH, bound=4.0):
    torch.manual_seed(SEED + 1)
    return (torch.rand(n, 2, dtype=torch.float64) * 2 * bound - bound)


class TestFloat64Invertibility:

    def test_f64_trained_invertibility(self, trained_spline_f64):
        z = _random_z_f64()
        x, _ = trained_spline_f64.forward(z)
        z_rec, _ = trained_spline_f64.inverse(x)
        err = (z - z_rec).abs().max().item()
        assert err < ATOL_F64, f"float64 z→x→z max err: {err:.2e}"

    def test_f64_trained_inverse_then_forward(self, trained_spline_f64):
        x = _random_input_f64()
        z, _ = trained_spline_f64.inverse(x)
        x_rec, _ = trained_spline_f64.forward(z)
        err = (x - x_rec).abs().max().item()
        assert err < ATOL_F64, f"float64 x→z→x max err: {err:.2e}"


class TestFloat64LogDetSymmetry:

    def test_f64_trained_logdet_cancel(self, trained_spline_f64):
        z = _random_z_f64()
        x, logdet_fwd = trained_spline_f64.forward(z)
        _, logdet_inv = trained_spline_f64.inverse(x)
        total = logdet_fwd + logdet_inv
        err = total.abs().max().item()
        assert err < ATOL_F64, f"float64 logdet_fwd + logdet_inv max err: {err:.2e}"


class TestFloat64LogProbConsistency:

    def test_f64_sample_logprob_matches(self, trained_spline_f64):
        torch.manual_seed(SEED + 20)
        z = torch.randn(BATCH, 2, dtype=torch.float64)
        x, logdet_fwd = trained_spline_f64.forward(z)
        log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)
        logprob_via_fwd = log_pz - logdet_fwd
        logprob_via_inv = trained_spline_f64.log_prob(x)
        err = (logprob_via_fwd - logprob_via_inv).abs().max().item()
        assert err < ATOL_F64, f"float64 sample vs log_prob max err: {err:.2e}"

    def test_f64_logdet_vs_autograd(self, trained_spline_f64):
        z = _random_z_f64(n=16)
        _, logdet_model = trained_spline_f64.forward(z)
        logdet_auto = _autograd_log_abs_det(trained_spline_f64.forward, z)
        err = (logdet_model - logdet_auto).abs().max().item()
        assert err < ATOL_F64, f"float64 logdet vs autograd max err: {err:.2e}"


class TestFloat64Normalization:

    def test_f64_density_integrates_to_one(self, trained_spline_f64):
        n_grid = 400
        bound = 6.0
        xs = torch.linspace(-bound, bound, n_grid, dtype=torch.float64)
        dx = xs[1] - xs[0]
        grid_x, grid_y = torch.meshgrid(xs, xs, indexing='ij')
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        with torch.no_grad():
            log_probs = []
            chunk = 4096
            for i in range(0, points.shape[0], chunk):
                log_probs.append(trained_spline_f64.log_prob(points[i:i+chunk]))
            log_probs = torch.cat(log_probs)

        integral = (log_probs.exp() * dx * dx).sum().item()
        # 0.967 级别的偏差来自 [-6,6]² 外的尾部截断，非精度问题
        assert abs(integral - 1.0) < 0.05, \
            f"float64 density integral = {integral:.6f}, expected ~1.0"
