import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from model.attention import TransformerLayer
from model.encoders import ImageTextEncoders, l2norm, ImageTextEncodersRecon
from model.decoder import MlpDecoder, TransformerDecoder

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.bn(x)
        x = rearrange(x, 'b d n -> b n d')
        x = F.relu(x)
        x = self.linear2(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, encoders: ImageTextEncoders, in_dim, args):
        super().__init__()
        self.encoders = encoders
        self.args = args
        self.amp = args.amp
        self.cma = TransformerLayer(
            query_dim=in_dim,
            ff_dim=in_dim,
            context_dim=in_dim,
            heads=args.cma_heads,
            dim_head=args.cma_head_dim,
            dropout=args.dropout,
            ff_activation='gelu',
            last_norm=True,
            last_fc=args.cma_last_fc,
            qk_norm=args.cma_qk_norm
        )

        self.cma_self = None
        if args.cma_self_attn:
            self.cma_self = TransformerLayer(
                query_dim=in_dim,
                ff_dim=in_dim,
                context_dim=None,
                heads=args.cma_heads,
                dim_head=args.cma_head_dim,
                dropout=args.dropout,
                ff_activation='gelu',
                last_norm=True,
                last_fc=False
            )

        self.last_mlp = None
        if args.cma_last_mlp:
            self.last_mlp = MLP(in_dim, in_dim // 2, in_dim)

        self.txt_l2 = args.info_txt_l2
        
    def forward(self, images, sentences, txt_len):
        with torch.cuda.amp.autocast(enabled=self.amp):
            img_emb, txt_emb, img_attn, txt_attn, img_residual, txt_residual, txt_bert =\
                self.encoders(images, sentences, txt_len)
            
            if self.cma_self is not None:
                img_emb = self.cma_self(img_emb)

            cm_feat = self.cma(
                repeat(txt_emb, 'b n d -> repeat b (n d)', repeat=img_emb.shape[0]), 
                context=img_emb
            )

            if self.last_mlp is not None:
                cm_feat = self.last_mlp(cm_feat)

            cm_feat = l2norm(cm_feat)
            
            if self.txt_l2:
                txt_emb = l2norm(txt_emb)
            
        return cm_feat, img_emb, txt_emb, img_attn, txt_attn, img_residual, txt_residual, txt_bert


class CrossModalAttentionRecon(nn.Module):
    def __init__(self, encoders: ImageTextEncodersRecon, in_dim, args):
        super().__init__()
        self.encoders = encoders
        self.args = args
        self.amp = args.amp
        self.cma = TransformerLayer(
            query_dim=in_dim,
            ff_dim=in_dim,
            context_dim=in_dim,
            heads=args.cma_heads,
            dim_head=args.cma_head_dim,
            dropout=args.dropout,
            ff_activation='gelu',
            last_norm=True,
            last_fc=args.cma_last_fc
        )
        
        self.cma_self = None
        if args.cma_self_attn:
            self.cma_self = TransformerLayer(
                query_dim=in_dim,
                ff_dim=in_dim,
                context_dim=None,
                heads=args.cma_heads,
                dim_head=args.cma_head_dim,
                dropout=args.dropout,
                ff_activation='gelu',
                last_norm=False,
                last_fc=False
            )

        self.last_mlp = None
        if args.cma_last_mlp:
            self.last_mlp = MLP(in_dim, in_dim // 2, in_dim)
        
        Decoder = {
            'mlp': MlpDecoder,
            'transformer': TransformerDecoder
        }[args.recon_decoder]

        self.decoder = Decoder(
            num_patches=int(args.crop_size/16),
            slot_dim=args.embed_dim,
            feat_dim=self.encoders.img_enc.local_feat_dim,
            normalizer=args.decoder_normalizer,
            self_attn=args.decoder_self_attn,
            pos_enc=args.decoder_pos_enc
        )

        self.txt_l2 = args.info_txt_l2

        self.mtp_mask_token_type = args.mtp_mask_token_type # ["concat", "add"]
        if self.mtp_mask_token_type == "concat":
            tpa_in_dim = in_dim + 1 # +1 for mask token
        else: 
            tpa_in_dim = in_dim

        self.tpa = None # token predicting attention
        self.mtp_tpa_type = args.mtp_tpa_type # {"self", "cross", +TODO}
        # if args.tpa_attn: # TODO
        if self.mtp_tpa_type == "self":
            self.tpa_self = TransformerLayer(
                query_dim=tpa_in_dim, 
                ff_dim=tpa_in_dim,
                context_dim=None,
                heads=args.cma_heads,
                dim_head=args.cma_head_dim,
                dropout=args.dropout,
                ff_activation='gelu',
                last_norm=True,
                last_fc=args.cma_last_fc
            )
        elif self.mtp_tpa_type == "cross":
            self.tpa_cross = TransformerLayer(
                query_dim=in_dim,
                ff_dim=in_dim,
                context_dim=in_dim,
                heads=args.cma_heads,
                dim_head=args.cma_head_dim,
                dropout=args.dropout,
                ff_activation='gelu',
                last_norm=True,
                last_fc=args.cma_last_fc
            )

        self.mtp_mask_idx_sample_mode = args.mtp_mask_idx_sample_mode # {"random", "txt_aff_topK", "ca_weights"}
        self.mtp_pred_target = args.mtp_pred_target # {"masked", "all"}
            
        
    def forward(self, images, sentences, txt_len):
        with torch.cuda.amp.autocast(enabled=self.amp):
            img_slot, img_feat, txt_emb, txt_attn, img_residual, txt_residual, txt_bert =\
                self.encoders(images, sentences, txt_len)
            img_feat_recon = self.decoder(img_slot)

            if self.cma_self is not None:
                img_slot = self.cma_self(img_slot)

            cm_feat = self.cma(
                repeat(txt_emb, 'b n d -> repeat b (n d)', repeat=img_slot.shape[0]), 
                context=img_slot
            )

            if self.last_mlp is not None:
                cm_feat = self.last_mlp(cm_feat)
            cm_feat = l2norm(cm_feat)

            if self.txt_l2:
                txt_emb = l2norm(txt_emb)
        
        # cm_feat: (B, B, D) = (32, 32, 512)
        # img_slot: (B, K, D) = (32, 36, 512)
        # txt_emb: (B, 1, D) = (32, 1, 512)
        # img_feat_recon: (B, N, D) = (32, 576, 512)
        # img_feat: (B, N, D) = (32, 576, 512)
        return cm_feat, img_slot, txt_emb, img_feat_recon, img_feat, txt_bert

    def masked_token_prediction(self, img_slot, txt_emb, cm_feat, r_mask=0.1, mask_type="zero", std=1):
        B, K, D = img_slot.shape
        
        # calculate n_mask
        n_mask = int(r_mask * img_slot.shape[1])

        # Reshape cm_feat
        cm_feat = cm_feat[0][:, None, :] # (B, B, D) -> (B, D) -> (B, 1, D) # TODO: cm_feat[0] or cm_feat[:, 0]

        # Normalize img_slot and txt_emb (norm == 1, following cm_feat)
        img_slot = img_slot / (torch.norm(img_slot ,dim=-1, keepdim=True) + 1e-6)
        txt_emb = txt_emb / (torch.norm(txt_emb ,dim=-1, keepdim=True) + 1e-6)

        # # Sample mask idx
        ''' DEPRECATED
        # # a. random 
        # if self.mtp_mask_idx_sample_mode.lower() == "random":
        #     mask_idx = torch.randperm(K)[:n_mask] # (n,)
        #     mask_idx = mask_idx[None, ...].repeat(B, 1) # (B, n)
        # # b. affinity topK
        # elif self.mtp_mask_idx_sample_mode.lower() == "txt_aff_topk":
        #     mask_idx = torch.topk(torch.sum(img_slot * txt_emb, dim=-1), k=n_mask, dim=-1).indices # (B, n)
        #     mask_idx_complement = torch.topk(-torch.sum(img_slot * txt_emb, dim=-1), k=K-n_mask, dim=-1).indices # (B, K-n)
        # # c. TODO: from cross-attention weights 
        # elif self.mtp_mask_idx_sample_mode.lower() == "ca_weights":
        #     pass
        '''
        # random shuffle
        # a. random 
        if self.mtp_mask_idx_sample_mode.lower() == "random":
            # batch_idx = torch.arange(B)[:, None]
            img_slot = img_slot[:, torch.randperm(K), :]
        # b. TODO: affinity topK
        if self.mtp_mask_idx_sample_mode.lower() == "txt_aff_topk":
            pass
        # c. TODO: from cross-attention weights 
        elif self.mtp_mask_idx_sample_mode.lower() == "ca_weights":
            pass

        # Make mix_tokens by concat 
        mix_tokens = img_slot # (B, K+2, D)
        # mix_tokens = torch.cat([img_slot, txt_emb, cm_feat], axis=1) # (B, K+2, D)
        
        # Set orig_img_slot
        orig_img_slot_mode = 0 # {0: "only masked", 1: "all"}
        if self.mtp_pred_target == "masked":
            orig_img_slot = mix_tokens[:, :n_mask, :] # (B, n, D)
        if self.mtp_pred_target == "all":
            orig_img_slot = mix_tokens[:, :-2, :]

        # Mask tokens
        mask_tokens = torch.zeros_like(mix_tokens[..., 0:1]) # (B, K+2, 1)
        # mask_tokens.requires_grad_(False) # TODO
        mask_tokens[:, :n_mask, :] = -1. # masked slot (from img_slot)
        mask_tokens[:, -2, :] = 1. # txt slot (txt_emb)
        mask_tokens[:, -1, :] = 2. # global slot (cm_feat)
        if mask_type == "zero":
            mix_tokens[:, :n_mask, :] = 0.
        elif mask_type == "noise" and std > 1e-4:
            std_tensor = torch.zeros(img_slot.shape[0], n_mask, img_slot.shape[2])
            std_tensor[...] = std
            contamination = torch.normal(mean=0, std=std_tensor).to(dtype=img_slot.dtype, device=img_slot.device)
            # contamination.requires_grad_(False) # TODO: 
            mix_tokens[:, :n_mask, :] = mix_tokens[:, :n_mask, :] + contamination
        
        if self.mtp_mask_token_type == "concat":
            mix_tokens = torch.cat([mix_tokens, mask_tokens], axis=-1) # (B, K+2, D+1)
        elif self.mtp_mask_token_type == "add":
            mix_tokens = mix_tokens + mask_tokens # (B, K+2, D)

        if self.mtp_tpa_type == "self":
            pred = self.tpa_self(mix_tokens)
        elif self.mtp_tpa_type == "cross":
            pred = self.tpa_cross(
                mix_tokens[:, :n_mask, :], 
                context=mix_tokens[:, n_mask:, :]
            )

        if self.mtp_pred_target == "masked":
            recon_img_slot = pred[:, :n_mask, :] # (B, n, D)
        if self.mtp_pred_target == "all":
            recon_img_slot = pred[:, :-2, :] # (B, K, D)
            
        if self.mtp_mask_token_type == "concat":
            recon_img_slot = recon_img_slot[..., :-1]

        recon_img_slot = recon_img_slot / (torch.norm(recon_img_slot ,dim=-1, keepdim=True) + 1e-6)

        return recon_img_slot, orig_img_slot