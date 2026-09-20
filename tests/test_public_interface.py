import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PublicInterfaceTest(unittest.TestCase):
    def test_patch_count_interface_is_unified(self):
        for relative in ("training/main_survival.py", "training/test_survival.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("'--num_patches'", source)
            self.assertIn("default=-1", source)
            self.assertNotIn("'--train_bag_size'", source)
            self.assertNotIn("'--val_bag_size'", source)
            self.assertNotIn("'--test_bag_size'", source)

    def test_sapa_spe_forward_backward(self):
        try:
            import torch
            from mil_models.model_otsurv import OTSurvPGSpatialPantherConcat
        except ImportError as exc:
            self.skipTest(f"optional runtime dependency unavailable: {exc}")

        model = OTSurvPGSpatialPantherConcat(
            patch_dim=1024,
            hidden_dim=256,
            num_prototypes=16,
        )
        features = torch.randn(32, 1024, requires_grad=True)
        coords = torch.rand(32, 2)
        output = model.forward_no_loss(
            [(features, coords), 10, 0],
            return_attn=True,
        )
        self.assertEqual(tuple(output["logits"].shape), (1, 1))
        self.assertEqual(tuple(output["Attn_OT"][0].shape), (32, 16))
        output["logits"].sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(bool(torch.isfinite(features.grad).all()))

    def test_optional_random_orthogonal_encoder(self):
        try:
            import torch
            from mil_models.model_otsurv import OTSurvPGSpatialPantherConcat
        except ImportError as exc:
            self.skipTest(f"optional runtime dependency unavailable: {exc}")

        model = OTSurvPGSpatialPantherConcat(
            patch_dim=32,
            hidden_dim=16,
            num_prototypes=4,
            patch_encoder_type="random_orthogonal",
            patch_encoder_seed=1,
        )
        self.assertFalse(any(p.requires_grad for p in model.patch_encoder.parameters()))
        features = torch.randn(12, 32, requires_grad=True)
        coords = torch.rand(12, 2)
        output = model.forward_no_loss([(features, coords), 10, 0])
        output["logits"].sum().backward()
        self.assertTrue(bool(torch.isfinite(features.grad).all()))


if __name__ == "__main__":
    unittest.main()
