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

        self.tpa_self = None # token predicting attention
        # if args.tpa_self_attn: # TODO
        if True:
            self.tpa_self = TransformerLayer(
                query_dim=in_dim+1, # +1 for mask token
                ff_dim=in_dim+1,
                context_dim=None,
                heads=args.cma_heads,
                dim_head=args.cma_head_dim,
                dropout=args.dropout,
                ff_activation='gelu',
                last_norm=False,
                last_fc=False
            )
        
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

    def masked_token_prediction(self, img_slot, txt_emb, cm_feat, n_mask=4, std=1):
        # Reshape cm_feat
        cm_feat = cm_feat[0][:, None, :] # (B, B, D) -> (B, D) -> (B, 1, D) # TODO: cm_feat[0] or cm_feat[:, 0]

        # Normalize img_slot and txt_emb (norm == 1, following cm_feat)
        img_slot = img_slot / torch.norm(img_slot ,dim=-1, keepdim=True)
        txt_emb = txt_emb / torch.norm(img_slot ,dim=-1, keepdim=True)

        # Sample mask idx
        sample_mode = "aff_topK" # ["random", "aff_topK", "ca_weights"]
        # a. random 
        if sample_mode.lower() == "random":
            mask_idx = torch.randperm(img_slot.shape[1])[:n_mask] 
        # b. affinity topK
        elif sample_mode.lower() == "aff_topk":
            mask_idx = torch.topk(torch.sum(img_slot * txt_emb, dim=-1), k=n_mask, dim=-1) # (B, K)
        # c. TODO: from cross-attention weights 
        elif sample_mode.lower() == "ca_weights":
            pass
        
        # Make mix_tokens by concat 
        mix_tokens = torch.cat([img_slot, txt_emb, cm_feat], axis=1) # (B, K+2, D)
        orig_img_slot = mix_tokens[:, mask_idx, :] # (B, n, D)
        
        # Mask tokens
        mask_tokens = torch.zeros_like(mix_tokens[..., 0:1]) # (B, K+2, 1)
        mask_tokens[:, mask_idx, :] = -1. # masked slot (from img_slot)
        mask_tokens[:, -2, :] = 1. # txt slot (txt_emb)
        mask_tokens[:, -1, :] = 2. # global slot (cm_feat)
        if std > 1e-4:
            std_tensor = torch.zeros(img_slot.shape[0], n_mask, img_slot.shape[2])
            std_tensor[...] = std
            contamination = torch.normal(mean=0, std=std_tensor).to(img_slot.device)
            contamination.requires_grad_(False) # TODO: is it right?
            mix_tokens[:, mask_idx, :] = mix_tokens[:, mask_idx, :] + contamination
        mix_tokens = torch.cat([mix_tokens, mask_tokens], axis=-1) # (B, K+1, D+1)

        recon_img_slot = self.tpa_self(mix_tokens)[:, mask_idx, :-1] # (B, n, D)

        return recon_img_slot, orig_img_slot