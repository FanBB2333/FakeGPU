"""Checkpoint state regression for the custom Phase 2 torch build."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch


def _run_scaled_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> torch.Tensor:
    x = torch.randn(8, 4, device="cuda")
    y = torch.randn(8, 2, device="cuda")
    with torch.amp.autocast(device_type="cuda"):
        loss = torch.nn.functional.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.detach()


def main() -> None:
    assert os.environ.get("TORCH_FAKEGPU_ENABLE") == "1"

    torch.cuda.manual_seed_all(1234)

    model = torch.nn.Linear(4, 2).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda")

    loss_before = _run_scaled_step(model, optimizer, scheduler, scaler)
    assert loss_before.device.type == "cuda"
    assert loss_before.is_cuda is True

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(checkpoint, f.name)
        loaded = torch.load(f.name, map_location="cuda:1")

    assert loaded["cuda_rng"][0].device.type == "cuda"
    assert loaded["cuda_rng"][0].device.index == 1
    assert loaded["cuda_rng"][0].is_cuda is True

    restored_model = torch.nn.Linear(4, 2).to("cuda")
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=1, gamma=0.5
    )
    restored_scaler = torch.amp.GradScaler("cuda")

    restored_model.load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_scheduler.load_state_dict(loaded["scheduler"])
    restored_scaler.load_state_dict(loaded["scaler"])
    torch.cuda.set_rng_state_all(loaded["cuda_rng"])

    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_scaler.get_scale() == scaler.get_scale()

    restored_state = restored_optimizer.state_dict()["state"]
    assert restored_state
    for param_state in restored_state.values():
        for value in param_state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cuda"
                assert value.is_cuda is True

    torch.cuda.set_rng_state_all(loaded["cuda_rng"])
    sample1 = torch.randn(4, 4, device="cuda")
    torch.cuda.set_rng_state_all(loaded["cuda_rng"])
    sample2 = torch.randn(4, 4, device="cuda")
    assert torch.equal(sample1.cpu(), sample2.cpu())

    loss_after = _run_scaled_step(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_scaler,
    )
    assert loss_after.device.type == "cuda"
    assert loss_after.is_cuda is True
    assert restored_scheduler.last_epoch == scheduler.last_epoch + 1
    assert next(restored_model.parameters()).device.type == "cuda"

    print("phase2 checkpoint state surface passed")


if __name__ == "__main__":
    main()
