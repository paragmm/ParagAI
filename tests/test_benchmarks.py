"""Unit tests for telemetry, diagnostics, and benchmark evaluation utilities."""

from paragai.utils.telemetry import TrainingTelemetry


def test_telemetry_initialization_and_update():
    tel = TrainingTelemetry(total_steps=1000)
    assert tel.total_steps == 1000
    assert tel.current_step == 0

    tel.update(step=10, loss=2.345, grad_norm=0.85, lr=3e-4, tokens_processed=512)
    assert tel.current_step == 10
    assert tel.train_loss == 2.345
    assert tel.grad_norm == 0.85
    assert tel.lr == 3e-4

    tel.log_eval(val_loss=2.123, val_ppl=8.35)
    assert tel.val_loss == 2.123
    assert tel.val_ppl == 8.35


def test_telemetry_dashboard_format():
    tel = TrainingTelemetry(total_steps=500)
    tel.update(step=50, loss=1.5, grad_norm=0.5, lr=1e-4, tokens_processed=1024)
    tel.log_eval(val_loss=1.6, val_ppl=4.95)

    dashboard = tel.format_dashboard()
    assert "ParagAI  Training  Dashboard" in dashboard
    assert "Step" in dashboard
    assert "Train Loss" in dashboard
    assert "Val Loss" in dashboard
    assert "Perplexity" in dashboard
