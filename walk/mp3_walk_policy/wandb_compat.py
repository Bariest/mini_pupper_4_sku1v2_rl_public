"""rsl_rl 5.0.1 starts W&B with ``wandb.Settings(start_method="thread")``, which wandb 0.30 rejects (the option was
removed). ``patch()`` drops it; call it before the runner is built when logging to W&B."""


def patch():
    import wandb
    settings = wandb.Settings
    if getattr(settings, "_mp3_patched", False):
        return

    def compat(*args, start_method=None, **kwargs):
        return settings(*args, **kwargs)
    compat._mp3_patched = True
    wandb.Settings = compat
