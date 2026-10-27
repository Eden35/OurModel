"""Behavioral checks for method equations, edge cases, and staged gradients."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from Models.evidence_guided_repair import EvidenceGuidedRepairNet, otsu_threshold
from utils.evidence_guided_losses import base_loss, outcome_loss, outcome_targets, repair_loss, signed_distance_target


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(71)

    def test_otsu_exact_and_ties(self):
        self.assertIsNone(otsu_threshold([3.0]))
        self.assertIsNone(otsu_threshold([0.5] * 4))
        self.assertEqual(otsu_threshold([0, 1, 2]), 0.5)
        self.assertEqual(otsu_threshold([0, 0, 2, 3]), 1.0)
        # Brute force reference with repeated values: each region is weighted equally.
        for _ in range(20):
            scores = np.random.RandomState(_).randint(0, 10, 13)
            unique = np.unique(scores)
            thresholds = (unique[:-1] + unique[1:]) / 2.0
            objectives = []
            for t in thresholds:
                a, b = scores[scores <= t], scores[scores > t]
                objectives.append(len(a) * len(b) * (a.mean() - b.mean()) ** 2)
            expected = thresholds[int(np.argmax(objectives))]
            self.assertAlmostEqual(otsu_threshold(scores), expected)

    def test_valid_pooling_and_replication_padding(self):
        model = EvidenceGuidedRepairNet()
        tensor = torch.arange(17 * 19).float().reshape(1, 1, 17, 19)
        regions = model._regions(17, 19, 16)
        pooled = model._pool_regions(tensor, regions)
        for i, (y, x, h, w) in enumerate(regions):
            self.assertAlmostEqual(pooled[0, i, 0].item(), tensor[0, 0, y:y+h, x:x+w].mean().item(), places=5)
        crop = model._crop(tensor, 16, 16, 1, 3, 16)
        self.assertTrue(torch.equal(crop[0, 0, 0, :3], tensor[0, 0, 16, 16:19]))
        self.assertTrue(torch.equal(crop[:, :, 0], crop[:, :, -1]))
        self.assertTrue(torch.equal(crop[:, :, :, 2], crop[:, :, :, -1]))

    def test_sdf_sign_scale_boundary_and_constant_masks(self):
        target = torch.zeros(3, 9, 11)
        target[0, :, 6:] = 1
        target[2] = 1
        distance, valid = signed_distance_target(target)
        self.assertEqual(valid.tolist(), [True, False, False])
        diagonal = math.sqrt(9 ** 2 + 11 ** 2)
        self.assertAlmostEqual(distance[0, 0, 4, 0].item(), 5 / diagonal)
        self.assertAlmostEqual(distance[0, 0, 4, -1].item(), -4 / diagonal)
        self.assertEqual(float(distance[0, 0, 4, 5]), 0.0)
        self.assertEqual(float(distance[0, 0, 4, 6]), 0.0)
        self.assertGreater(float(distance[0, 0, 0, 0]), 0.0)  # image edge is not an invented boundary

    def test_selection_uses_three_scores_and_excludes_js(self):
        model = EvidenceGuidedRepairNet()
        shape = (1, 1, 16, 48)
        entropy_s = torch.zeros(shape); entropy_s[:, :, :, 16:32] = 1
        entropy_c = torch.zeros(shape); entropy_c[:, :, :, 32:] = 4
        base = {"entropy_s": entropy_s, "entropy_c": entropy_c, "sdf_sigma": torch.ones(shape),
                "initial_logits": torch.zeros(1, 2, 16, 48), "js_divergence": torch.ones(shape) * 10}
        self.assertEqual(model._select_regions(base, model._regions(16, 48, 16)), [[1, 2]])

    def test_empty_selection_skips_all_extra_modules(self):
        model = EvidenceGuidedRepairNet().eval()
        calls = []
        handles = [module.register_forward_hook(lambda *args: calls.append(True)) for module in
                   (model.token_encoder, model.query, model.key, model.value, model.repair, model.outcome)]
        with torch.no_grad():
            out = model(torch.randn(1, 3, 15, 13), return_details=True)
        self.assertEqual(calls, [])
        self.assertFalse(out["selected_mask"].any())
        self.assertTrue(torch.equal(out["final_probabilities"], out["probabilities_0"]))
        for handle in handles:
            handle.remove()

    def test_attention_and_batched_zero_residual(self):
        model = EvidenceGuidedRepairNet().eval()
        repair_batch_sizes = []
        handle = model.repair.register_forward_hook(lambda module, args, out: repair_batch_sizes.append(args[0].size(0)))
        with torch.no_grad():
            out = model(torch.randn(2, 3, 65, 81), return_details=True)
        self.assertEqual(repair_batch_sizes, [int(out["selected_mask"].sum())])
        for batch, ids, weights in out["reference_attention"]:
            self.assertTrue(torch.equal(weights[torch.arange(len(ids)), ids], torch.zeros(len(ids))))
            self.assertTrue(torch.allclose(weights.sum(1), torch.ones(len(ids))))
            self.assertEqual(weights.size(1), len(out["regions"]))
        self.assertTrue(torch.allclose(out["candidate_probabilities"], out["probabilities_0"], atol=1e-6))
        self.assertTrue(torch.allclose(out["final_probabilities"].sum(1), torch.ones(2, 65, 81), atol=1e-6))
        handle.remove()

    def test_router_accepts_strict_positive_gain_only(self):
        model = EvidenceGuidedRepairNet().eval()
        with torch.no_grad():
            model.repair.residual.bias.copy_(torch.tensor([-2., 2.]))
        image = torch.randn(1, 3, 65, 81)
        for values, accept in [([0.8, 0.2], True), ([0.2, 0.8], False), ([0.5, 0.5], False)]:
            with patch.object(model.outcome, 'forward', side_effect=lambda x, v=values: x.new_tensor(v).expand(x.size(0), 2)):
                with torch.no_grad():
                    out = model(image, return_details=True)
            self.assertEqual(int(out['accepted_mask'].sum()), int(out['selected_mask'].sum()) if accept else 0)
            self.assertTrue(torch.equal(out['final_probabilities'], out['candidate_probabilities'] if accept else out['probabilities_0']))
            # Unselected spatial regions are never modified even when candidates are accepted.
            for i, (y, x, h, w) in enumerate(out['regions']):
                if not out['selected_mask'][0, i]:
                    self.assertTrue(torch.equal(out['candidate_probabilities'][0, :, y:y+h, x:x+w], out['probabilities_0'][0, :, y:y+h, x:x+w]))

    def test_gain_harm_targets_and_per_image_weighting(self):
        # Two images; unequal selected-region counts, with an incomplete final region.
        truth = torch.tensor([[[0, 0, 1]], [[0, 1, 1]]])
        initial = torch.tensor([[[1, 0, 0]], [[0, 1, 1]]])
        candidate = torch.tensor([[[0, 1, 1]], [[0, 1, 0]]])
        out = {'probabilities_0': F.one_hot(initial, 2).permute(0, 3, 1, 2).float(),
               'candidate_probabilities': F.one_hot(candidate, 2).permute(0, 3, 1, 2).float(),
               'regions': [(0, 0, 1, 2), (0, 2, 1, 1)],
               'selected_mask': torch.tensor([[True, True], [False, True]]),
               'outcome': torch.zeros(2, 2, 2)}
        gain, harm = outcome_targets(out, truth)
        self.assertEqual(gain.tolist(), [0.5, 1., 0.])
        self.assertEqual(harm.tolist(), [0.5, 0., 1.])
        self.assertAlmostEqual(float(outcome_loss(out, truth)), (0.75 + 1.) / 2)
        error_delta = ((candidate == truth).float() - (initial == truth).float())
        self.assertAlmostEqual(float(gain[0] - harm[0]), float(error_delta[0, :, :2].mean()))

    def test_full_image_repair_loss_and_no_selection_exclusion(self):
        logits = torch.randn(2, 2, 2, 4, requires_grad=True)
        truth = torch.tensor([[[0, 0, 1, 1], [0, 1, 0, 1]], [[0, 1, 0, 1], [1, 0, 1, 0]]])
        probabilities = logits.softmax(1)
        out = {'candidate_logits': logits, 'candidate_probabilities': probabilities,
               'selected_mask': torch.tensor([[True, False], [False, False]])}
        p = probabilities[0, 1]; y = truth[0].float()
        expected = F.cross_entropy(logits[:1], truth[:1]) + 1 - (2 * (p*y).sum() + 1e-6) / (p.sum()+y.sum()+1e-6)
        self.assertTrue(torch.allclose(repair_loss(out, truth), expected))
        repair_loss(out, truth).backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)

    def test_all_stages_backprop_and_freeze_statistics(self):
        model = EvidenceGuidedRepairNet()
        image = torch.randn(2, 3, 64, 80)
        truth = torch.randint(0, 2, (2, 64, 80))
        for stage in (1, 2, 3):
            model.zero_grad(set_to_none=True)
            model.set_stage(stage).train()
            frozen = {name: value.clone() for name, value in model.state_dict().items()
                      if name.split('.')[0] in [key for key, module in model.named_children() if not any(p.requires_grad for p in module.parameters())]}
            out = model(image, accept=False, return_details=True)
            loss = base_loss(out, truth)['total'] if stage == 1 else repair_loss(out, truth) if stage == 2 else outcome_loss(out, truth)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            active_grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None)
            self.assertGreater(active_grad, 0.)
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    self.assertIsNone(parameter.grad)
            for name, value in frozen.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
