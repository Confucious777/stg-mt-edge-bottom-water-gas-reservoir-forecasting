import unittest

import torch

from models.time_modeling import (
    MultiScaleTemporalModule,
    PhysicsGuidance,
    ResidualDecomposition,
    TransformerEncoder,
    Upsample,
)


class TestTimeModeling(unittest.TestCase):
    def test_upsample_and_repeat_shape(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 65, 3, 4)
        mask = torch.ones(2, 65, 3)
        mask[:, 60:, 1] = 0.0
        month_index = torch.cat(
            [
                torch.full((31,), 2024 * 12 + 1),
                torch.full((28,), 2024 * 12 + 2),
                torch.full((6,), 2024 * 12 + 3),
            ]
        ).unsqueeze(0).repeat(2, 1)

        upsample = Upsample(keep_incomplete=True)
        x_m, mask_m, day_to_month = upsample(x, mask, month_index=month_index)
        self.assertEqual(x_m.shape, (2, 3, 3, 4))
        self.assertEqual(mask_m.shape, (2, 3, 3))
        self.assertEqual(day_to_month.shape, (2, 65))

        x_d = Upsample.repeat_to_daily(x_m, day_to_month=day_to_month, t_day=65)
        self.assertEqual(x_d.shape, (2, 65, 3, 4))

    def test_transformer_encoder_mask(self) -> None:
        torch.manual_seed(1)
        encoder = TransformerEncoder(
            input_dim=5,
            d_model=16,
            num_heads=4,
            num_layers=1,
            d_ff=32,
            dropout=0.0,
        )
        x = torch.randn(1, 12, 2, 5)
        mask = torch.ones(1, 12, 2)
        mask[:, :, 1] = 0.0

        out = encoder(x, mask=mask)
        self.assertEqual(out.shape, (1, 12, 2, 16))
        self.assertTrue(torch.allclose(out[:, :, 1, :], torch.zeros_like(out[:, :, 1, :]), atol=1e-6))

    def test_physics_guidance_masked_loss(self) -> None:
        torch.manual_seed(2)
        physics = PhysicsGuidance(padding_value=-999.0)
        x = torch.rand(1, 8, 3, 4)
        node_mask = torch.ones(1, 8, 3)
        influx = torch.rand(1, 8, 3)
        influx_mask = torch.ones(1, 8, 3)
        influx_mask[:, 6:, :] = 0.0

        fused = physics.fuse_features(x, influx_seq=influx, node_mask=node_mask, influx_mask=influx_mask)
        self.assertEqual(fused.shape, (1, 8, 3, 5))
        self.assertTrue(torch.allclose(fused[:, 6:, :, -1], torch.zeros_like(fused[:, 6:, :, -1]), atol=1e-6))

        pred = influx.clone()
        pred[influx_mask > 0.5] = pred[influx_mask > 0.5] + 1.0
        loss = physics.masked_mse(pred=pred, target=influx, mask=influx_mask)
        self.assertAlmostEqual(float(loss.item()), 1.0, places=5)

    def test_residual_decomposition_shape(self) -> None:
        torch.manual_seed(3)
        module = ResidualDecomposition(input_dim=10, hidden_dim=16, num_blocks=2, pred_dim=1, dropout=0.0)
        x = torch.randn(2, 7, 4, 10)
        mask = torch.ones(2, 7, 4)
        y, hidden, states = module(x, mask=mask, return_states=True)
        self.assertEqual(y.shape, (2, 7, 4, 1))
        self.assertEqual(hidden.shape, (2, 7, 4, 10))
        self.assertIn("block_pred", states)
        self.assertEqual(states["block_pred"].shape[0], 2)

    def test_multiscale_temporal_forward(self) -> None:
        torch.manual_seed(4)
        module = MultiScaleTemporalModule(
            input_dim=4,
            d_model=12,
            num_heads=3,
            num_layers=1,
            d_ff=24,
            dropout=0.0,
            keep_incomplete_month=True,
            fuse_mode="add",
            padding_value=-999.0,
        )
        x = torch.rand(2, 40, 5, 4)
        node_mask = torch.ones(2, 40, 5)
        node_mask[:, :2, 0] = 0.0
        month_index = torch.cat(
            [
                torch.full((20,), 2024 * 12 + 1),
                torch.full((20,), 2024 * 12 + 2),
            ]
        ).unsqueeze(0).repeat(2, 1)
        influx_seq = torch.rand(2, 40, 5)
        influx_mask = torch.ones(2, 40, 5)
        influx_mask[:, :2, 0] = 0.0
        out = module(
            x,
            node_mask=node_mask,
            month_index=month_index,
            influx_seq=influx_seq,
            influx_mask=influx_mask,
        )

        self.assertEqual(out["h_time"].shape, (2, 40, 5, 12))
        self.assertEqual(out["h_distant_month"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
