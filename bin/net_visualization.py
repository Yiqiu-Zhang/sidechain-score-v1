import torch.nn as nn
from torchsummary import summary

def vis_model():
   model = modelling.BertForDiffusion(
        config=cfg,
        time_encoding=time_encoding,
        decoder=decoder,
        ft_is_angular=dsets[0].dset.feature_is_angular[ft_key],
        ft_names=dsets[0].dset.feature_names[ft_key],
        lr=lr,
        loss=loss_fn,
        use_pairwise_dist_loss=use_pdist_loss
        if isinstance(use_pdist_loss, float)
        else [*use_pdist_loss, timesteps],
        l2=l2_norm,
        l1=l1_norm,
        circle_reg=circle_reg,
        epochs=max_epochs,
        steps_per_epoch=len(train_dataloader),
        lr_scheduler=lr_scheduler,
        num_encoder_layers=num_encoder_layers,
        write_preds_to_dir=results_folder / "valid_preds"
        if write_valid_preds
        else None,
    )

