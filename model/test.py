import dataset
from rigid_diffusion_score import RigidDiffusion
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, DataLoader, lightning

full_data_name = 'cath_test.pkl'
loaders = []
datasets = []

def train():

    transform = dataset.TorsionNoiseTransform()
    dsets = [dataset.ProteinDataset(split=s,
                                    pickle_dir=full_data_name,
                                    transform=transform) for s in ('train', 'val')]

    datamodule = lightning.LightningDataset(train_dataset=dsets[0], val_dataset=dsets[1])

    model = modelling.AngleDiffusion(
        config=cfg,
        time_encoding=time_encoding,
        pred_all=pred_all,
        # decoder=decoder,
        ft_is_angular=dsets[0].dset.feature_is_angular[ft_key],
        ft_names=dsets[0].dset.feature_names[ft_key],
        lr=lr,
        loss=loss_fn,
        #  diffusion_fraction = 0.7,
        use_pairwise_dist_loss=use_pdist_loss
        if isinstance(use_pdist_loss, float)
        else [*use_pdist_loss, timesteps],
        l2=l2_norm,
        l1=l1_norm,
        circle_reg=circle_reg,
        epochs=max_epochs,
        steps_per_epoch=len(train_dataloader),
        lr_scheduler=lr_scheduler,
        # num_encoder_layers=num_encoder_layers,
        write_preds_to_dir=results_folder / "valid_preds"
        if write_valid_preds
        else None,
    )
    cfg.save_pretrained(results_folder)

    callbacks = build_callbacks(outdir=results_folder)

    # Get accelerator and distributed strategy
    accelerator, strategy = "cpu", None
    if not cpu_only and torch.cuda.is_available():
        accelerator = "cuda"
        if torch.cuda.device_count() > 1:
            # https://github.com/Lightning-AI/lightning/discussions/6761https://github.com/Lightning-AI/lightning/discussions/6761
            strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        default_root_dir=results_folder,
        gradient_clip_val=gradient_clip,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger(save_dir=results_folder / "logs"),
        log_every_n_steps=len(train_dataloader),  # Log >= once per epoch
        accelerator=accelerator,
        strategy=strategy,
        gpus=ngpu,
        enable_progress_bar=False,
        move_metrics_to_cpu=False,  # Saves memory
       # detect_anomaly=True
     #   profiler=profiler,
      #  amp_backend='apex',
     #   amp_level = 'O1'
    )

    trainer.fit(model, datamodule)
