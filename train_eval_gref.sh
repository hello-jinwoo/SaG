python train_cma_recon.py \
--remark mtp_gref_poc67 \
--mtp_mask_token_type concat \
--mtp_mask_idx_sample_mode txt_aff_topk \
--mtp_init_epoch 0 \
--mtp_loss_weight 0.5 \
--mtp_mask_type zero \
--mtp_mask_ratio 0.5 \
--mtp_pred_target masked \
--mtp_tpa_type cross \
--data_name coco \
--margin 0.1 \
--img_num_embeds 36 \
--txt_num_embeds 1 \
--img_finetune \
--txt_finetune \
--batch_size 32 \
--warm_epoch 0 \
--num_epochs 50 \
--optimizer adamw \
--lr_scheduler cosine \
--lr_step_size 20 \
--lr_step_gamma 0.1 \
--warm_img \
--log_step 200 \
--agg_pos_enc sine \
--lr 1e-5 \
--txt_lr_scale 1 \
--img_spm_lr_scale 1 \
--img_lr_scale 0.1 \
--eval_on_gpu \
--sync_bn \
--amp \
--agg_activation gelu \
--agg_query_self_attns 0 \
--agg_latent_head 4 \
--agg_latent_dim 128 \
--agg_ff_mult 4 \
--agg_last_fc \
--agg_last_ln \
--agg_pre_norm \
--agg_cross_head 4 \
--agg_cross_dim 256 \
--agg_depth 6 \
--agg_weight_sharing \
--agg_input_dim 512 \
--agg_query_dim 512 \
--embed_dim 512 \
--agg_self_per_cross_attn 1 \
--recon_weight 1 \
--agg_query_type entity \
--cascade_factor 3 \
--data_path ./data/refcoco/Gref \
--agg_cross_attn_type slot \
--workers 4 \
--grad_clip 0.1 \
--dropout 0.1 \
--txt_pooling cls \
--weight_decay 1e-4 \
--crop_size 384 \
--pseudo_threshold 0.5 \
--cma_heads 4 \
--cma_head_dim 256 \
--cma_criterion info_nce \
--cma_detach_target \
--lr_warmup_iter 5000 \
--seed 1 \
--num_layers 12 \
--agg_var_scaling 1 \
--decoder_normalizer softmax


python eval_cma_recon.py \
--remark mtp_gref_debug \
--data_name coco \
--margin 0.1 \
--img_num_embeds 36 \
--txt_num_embeds 1 \
--img_finetune \
--txt_finetune \
--batch_size 32 \
--warm_epoch 0 \
--num_epochs 50 \
--optimizer adamw \
--lr_scheduler cosine \
--lr_step_size 20 \
--lr_step_gamma 0.1 \
--warm_img \
--log_step 200 \
--agg_pos_enc sine \
--lr 1e-5 \
--txt_lr_scale 1 \
--img_spm_lr_scale 1 \
--img_lr_scale 0.1 \
--eval_on_gpu \
--sync_bn \
--amp \
--agg_activation gelu \
--agg_query_self_attns 0 \
--agg_latent_head 4 \
--agg_latent_dim 128 \
--agg_ff_mult 4 \
--agg_last_fc \
--agg_last_ln \
--agg_pre_norm \
--agg_cross_head 4 \
--agg_cross_dim 256 \
--agg_depth 6 \
--agg_weight_sharing \
--agg_input_dim 512 \
--agg_query_dim 512 \
--embed_dim 512 \
--agg_self_per_cross_attn 1 \
--recon_weight 1 \
--agg_query_type entity \
--cascade_factor 3 \
--data_path ./data/refcoco/Gref \
--agg_cross_attn_type slot \
--workers 4 \
--grad_clip 0.1 \
--dropout 0.1 \
--txt_pooling cls \
--weight_decay 1e-4 \
--crop_size 384 \
--pseudo_threshold 0.5 \
--cma_heads 4 \
--cma_head_dim 256 \
--cma_criterion info_nce \
--cma_detach_target \
--lr_warmup_iter 5000 \
--seed 1 \
--num_layers 12 \
--agg_var_scaling 1 \
--decoder_normalizer softmax 
