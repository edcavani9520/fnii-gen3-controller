    # ═══════════════════════════════════════════════════════════════
    # Kinova Gen3 π0.5 LoRA fine-tuning config
    # Dataset: 100 episodes, 7-dim delta action, 8-dim state
    # Hardware: 2x RTX 5090 (32GB each)
    # ═══════════════════════════════════════════════════════════════
    TrainConfig(
        name="pi05_kinova_lora",
        # ----- Model -----
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=12,                 # 12 frames = 1.2s @ 10Hz
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",  # LoRA on PaliGemma
            action_expert_variant="gemma_300m_lora",  # LoRA on Action Expert
        ),
        # ----- Data -----
        data=LeRobotLiberoDataConfig(
            repo_id="kinova_cube",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ).create,
        # ----- Weight Loading -----
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # ----- LoRA freeze filter -----
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=12,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # ----- Training -----
        num_train_steps=30_000,
        batch_size=32,                          # per GPU, 2 GPUs = 64
        # ----- Optimizer -----
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,                         # LoRA不需要EMA
        # ----- Save & Log -----
        save_interval=2_500,
        log_interval=50,
        keep_period=10_000,
        overwrite=True,
        num_workers=8,
        # ----- Wandb -----
        wandb_enabled=True,
        wandb_project="pi05_kinova",
    ),