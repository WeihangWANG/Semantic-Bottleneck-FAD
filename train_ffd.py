import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys

sys.path.append("..")
import argparse
from typing import Any, Dict, List, Optional
from timm.utils import accuracy, AverageMeter
import time
import argparse
import datetime
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timm import create_model

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils_st import load_checkpoint, save_checkpoint, save_max_checkpoint, get_grad_norm, auto_resume_helper, \
    reduce_tensor
from performance import performances_val

# -----------------------------
# New: cosine alignment loss + model wrapper
# -----------------------------
def cosine_align_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a, b: (B, D)
    """
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (1.0 - (a * b).sum(dim=-1)).mean()

class TokenPool(nn.Module):
    def __init__(self, dim=2560, num_queries=64, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, tokens, valid_mask=None):
        # 统一 dtype：跟 attn 权重一致（通常 float32）
        dtype = self.attn.in_proj_weight.dtype
        tokens = tokens.to(dtype)
        q = self.queries.to(dtype).expand(tokens.size(0), -1, -1)

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask  # bool

        out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        out = self.ln(out)
        return out

def select_topk_tokens_by_l2(tokens_pad: torch.Tensor,
                            valid_mask: torch.Tensor,
                            k_prime: int = 64):
    """
    tokens_pad: (B, Mmax, D)  (teacher padded tokens)
    valid_mask: (B, Mmax)     True=valid token, False=padding
    return:
      sel_tokens: (B, k_prime, D)   （不足 k_prime 的样本会被 0 padding）
      sel_mask:   (B, k_prime)      True=valid selected, False=padding
    """
    B, Mmax, D = tokens_pad.shape
    device = tokens_pad.device

    # 分数：L2 norm（只对 valid token 有效；padding 设为 -inf）
    scores = tokens_pad.float().norm(dim=-1)  # (B,Mmax)
    scores = scores.masked_fill(~valid_mask, float("-inf"))

    sel_tokens = tokens_pad.new_zeros((B, k_prime, D))            # same dtype as tokens_pad
    sel_mask = torch.zeros((B, k_prime), device=device, dtype=torch.bool)

    # 每个样本独立 top-k，避免某些样本 valid_len < k_prime 导致 topk 选到 padding
    valid_lens = valid_mask.sum(dim=1).tolist()
    for b in range(B):
        k_b = min(k_prime, int(valid_lens[b]))
        if k_b <= 0:
            continue
        idx = torch.topk(scores[b], k=k_b, dim=0, largest=True).indices  # (k_b,)
        sel_tokens[b, :k_b] = tokens_pad[b, idx]                         # gather tokens
        sel_mask[b, :k_b] = True

    return sel_tokens, sel_mask

def get_vis_w(epoch: int,
              step_idx: int,
              num_steps: int,
              vis_w_max: float,
              start_epoch: int = 2,
              warmup_epochs: float = 2.0):
    """
    vis_w 动态调度：
      - epoch < start_epoch: 0
      - start_epoch 起，经过 warmup_epochs 线性升到 vis_w_max
      - 之后保持 vis_w_max
    用 epoch+step 的连续时间更平滑。
    """
    t = epoch + (step_idx / max(1, num_steps))  # continuous
    if t < start_epoch:
        return 0.0
    prog = (t - start_epoch) / max(1e-6, warmup_epochs)
    prog = min(max(prog, 0.0), 1.0)
    return vis_w_max * prog

def token_cosine_loss(a, b, eps=1e-8):
    # a,b: (B,K,D)
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (1.0 - (a * b).sum(dim=-1)).mean()

def relation_distill_loss(teacher, student, tau=0.07, mode="mse"):
    teacher = F.normalize(teacher.float(), dim=-1, eps=1e-6)
    student = F.normalize(student.float(), dim=-1, eps=1e-6)
    t_sim = (teacher @ teacher.transpose(1, 2)) / tau
    s_sim = (student @ student.transpose(1, 2)) / tau
    if mode == "mse":
        return F.mse_loss(s_sim, t_sim.detach())
    elif mode == "kl":
        t_prob = F.softmax(t_sim, dim=-1).detach()
        s_logp = F.log_softmax(s_sim, dim=-1)
        return F.kl_div(s_logp, t_prob, reduction="batchmean")
    else:
        raise ValueError


def parse_block_list(s: str) -> List[int]:
    return sorted({int(x) for x in s.split(',') if x.strip().isdigit()}) if s else []

def freeze_vit_except_blocks(vit, unfreeze_first=2, unfreeze_last=2,
                             train_patch_embed=False, train_norm=False,
                             train_pos_embed=False, train_cls_token=False):
    """
    vit: timm VisionTransformer
    只解冻 vit.blocks 的前 unfreeze_first 个、后 unfreeze_last 个 block。
    其余参数全部冻结。
    可选是否训练 patch_embed / norm / pos_embed / cls_token
    """
    # 1) freeze all
    for p in vit.parameters():
        p.requires_grad = False

    # 2) unfreeze selected transformer blocks
    n_blocks = len(vit.blocks)
    for i, blk in enumerate(vit.blocks):
        if i < unfreeze_first or i >= (n_blocks - unfreeze_last):
            for p in blk.parameters():
                p.requires_grad = True

    # 3) optional parts
    if train_patch_embed and hasattr(vit, "patch_embed"):
        for p in vit.patch_embed.parameters():
            p.requires_grad = True

    if train_norm and hasattr(vit, "norm"):
        for p in vit.norm.parameters():
            p.requires_grad = True

    # pos_embed / cls_token are usually nn.Parameter on the module
    if train_pos_embed and hasattr(vit, "pos_embed") and vit.pos_embed is not None:
        vit.pos_embed.requires_grad = True

    if train_cls_token and hasattr(vit, "cls_token") and vit.cls_token is not None:
        vit.cls_token.requires_grad = True


class ViTVideoStudent(nn.Module):
    """
    Robust design:
      - Classification uses ViT video-level feature (768) -> MLP head -> 2 classes
      - Semantic alignment uses student projection: 768 -> 2560, only for feature alignment loss
    """
    def __init__(self, vit_backbone: nn.Module, sem_dim: int = 2560, num_classes: int = 2, unfreeze_blocks = "0,1,10,11",
                 cls_mlp_hidden: int = 256, cls_dropout: float = 0.1, vis_num_queries: int = 16, vis_num_heads: int = 8):
        super().__init__()
        self.vit = vit_backbone

        feat_dim = self.vit.head.in_features  # ViT-B/16 is 768
        # Do NOT use the original classifier head for forward; keep backbone feature only
        self.vit.head = nn.Identity()

        # Classification head on 768 (stable)
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, cls_mlp_hidden),
            nn.GELU(),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_mlp_hidden, num_classes),
        )

        # Student-side projection for semantic alignment: 768 -> 2560
        self.sem_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, sem_dim),
        )

        # # 视觉 token 蒸馏：学生 patch token 196×768 -> 196×2560
        # self.vis_proj = nn.Sequential(
        #     nn.LayerNorm(feat_dim),
        #     nn.Linear(feat_dim, sem_dim),
        # )

        # # token pooling（学生/教师共用一套 pool，输出固定 K×2560）
        # self.vis_pool = TokenPool(dim=sem_dim, num_queries=vis_num_queries, num_heads=vis_num_heads)

    def forward(self, x: torch.Tensor):
        """
        x: (N,3,224,224), N=B*T
        return:
          logits: (N,2)
          cls_feat: (N,768)
          patch_feat: (N,196,768)
        """
        feat = self.vit.forward_features(x)  # 多数 timm: (N,197,768)

        if feat.dim() == 3:
            cls_feat = feat[:, 0, :]      # (N,768)
            # patch_feat = feat[:, 1:, :]   # (N,196,768)
        else:
            raise RuntimeError(f"Unexpected vit.forward_features shape: {feat.shape}")

        logits = self.cls_head(cls_feat)  # (N,2)
        return logits, cls_feat#, patch_feat

def parse_option():
    parser = argparse.ArgumentParser('ViT-FAS training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--frame_len', type=int, help="frame length for single video")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp_opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--sem_w', type=float, default=0.5)
    parser.add_argument('--vis_w', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='cdf')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    reduction = "mean" if is_average else "none"
    return F.cross_entropy(logit, truth, reduction=reduction)


def main(config):
    val_name = 'ocim'
    data_loader_train, data_loader_val = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    # model = build_model(config)
    vit = create_model('vit_base_patch16_224.augreg_in21k', pretrained=False)
    # load local pretrained file
    state = torch.load('vit.bin', map_location='cpu')
    vit.load_state_dict(state, strict=True)
    ## 冻结 ViT，只解冻前2后2个 blocks
    # freeze_vit_except_blocks(
    #     vit,
    #     unfreeze_first=2,
    #     unfreeze_last=2,
    #     train_patch_embed=False,
    #     train_norm=True,
    #     train_pos_embed=False,
    #     train_cls_token=False,
    # )

    model = ViTVideoStudent(vit_backbone=vit, sem_dim=2560, num_classes=2, cls_mlp_hidden=128,
                            cls_dropout=0.1)

    # model.head = nn.Linear(model.head.in_features, 2)

    model.cuda()
    # logger.info(str(model))

    optimizer = build_optimizer(config, model)

    model = torch.nn.DataParallel(model)

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = softmax_cross_entropy_criterion

    max_accuracy = 0.5
    max_epoch = 0
    sem_w = config.SEM_W
    vis_w = config.VIS_W

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        hter, auc, acc = validate(config, data_loader_val, model, val_name)
        logger.info(f"HTER of the network on the oulu test images: {hter:.4f} with AUC: {auc:.4f} and ACC: {acc:.4f}.")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, sem_w, vis_w)

        hter, auc, acc = validate(config, data_loader_val, model, val_name)
        logger.info(f"HTER of the network on the test images: {hter:.4f} with AUC: {auc:.4f} and ACC: {acc:.4f}.")

        # if hter < max_accuracy:
        #     max_accuracy = hter
        #     max_epoch = epoch
        #     save_max_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        if auc > max_accuracy:
            max_accuracy = auc
            max_epoch = epoch
            save_max_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        # max_accuracy = min(max_accuracy, hter)
        max_accuracy = max(max_accuracy, auc)
        logger.info(f"Minimal HTER of the network on the test images: {max_accuracy:.4f} with epoch #{max_epoch}.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, sem_w, vis_w):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    B = config.DATA.BATCH_SIZE
    T = config.DATA.FRAME_LEN

    # for idx, (samples, targets) in enumerate(data_loader):
    for idx, (samples, targets, semantic_token) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        samples = samples.view(-1, samples.size(2), samples.size(3), samples.size(4))
        targets = targets.cuda(non_blocking=True)
        semantic_token = semantic_token.cuda(non_blocking=True)
        semantic_token = semantic_token.view(-1, semantic_token.size(2))

        # ----- forward: get frame logits + frame features -----
        logits_frame, feat_frame = model(samples)  # logits: (B*T,2), feat:(B*T,768)
        targets = targets.view(-1)
        loss_ce = criterion(logits_frame, targets)

        # ----- frame-level features for semantic alignment -----
        # semantic teacher embedding: make it (B,2560)
        if semantic_token.dim() == 3:
            sem_teacher = semantic_token.mean(dim=1)  # (B,2560)
        elif semantic_token.dim() == 2:
            sem_teacher = semantic_token
        else:
            raise ValueError(f"Unexpected semantic_token shape: {tuple(semantic_token.shape)}")

        # student projection 768->2560 (under DataParallel use model.module)
        sem_student = model.module.sem_proj(feat_frame)  # (B,2560)

        # alignment loss (detach teacher)
        loss_align = cosine_align_loss(sem_student, sem_teacher.detach())

        # total loss
        # loss = loss_ce
        loss = loss_align
        # loss = loss_ce + sem_w * loss_align# + cur_vis_w * loss_vis

        # ----- backward / optim (keep your original accumulation logic) -----
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                # if config.TRAIN.CLIP_GRAD:
                #     grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                # else:
                #     grad_norm = get_grad_norm(amp.master_params(optimizer))
                raise NotImplementedError("AMP (apex) is commented out in your original code; enable it if needed.")
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.avg:.4f}\t'
                f'loss {loss_meter.avg:.4f}\t' 
                # f"loss_cls {loss_ce.item():.4f} loss_sem {loss_align.item():.4f}\t"
                # f"loss_vis {loss_vis.item():.4f}\t"
                f'grad_norm {norm_meter.avg:.4f}\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, name):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = softmax_cross_entropy_criterion

    model.eval()
    with torch.no_grad():
        score_list = []  # video-level: "prob label"
        score_frame_list = []  # frame-level: "prob label"

        for idx, (samples, target) in enumerate(data_loader):
            samples = samples.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # 支持 [B,F,C,H,W]（视频）或 [B,C,H,W]（单帧）
            if samples.dim() == 5:
                B, F, C, H, W = samples.shape
                flat = samples.view(B * F, C, H, W)
                logits,_ = model(flat)  # [B*F, num_classes]
                probs = torch.softmax(logits, dim=1)[:, 1].view(B, F)  # [B,F]，取第1类概率
            else:
                B, C, H, W = samples.shape
                F = 1
                logits,_ = model(flat)  # [B, num_classes]
                probs = torch.softmax(logits, dim=1)[:, 1].view(B, 1)  # [B,1]

            # 适配 target 形状：支持 [B]、[B,1]、[B,F] 等
            if target.dim() == 1:  # [B]
                tgt_vid = target.long()  # [B]
                tgt_frame = tgt_vid.view(B, 1).repeat(1, F)
            elif target.dim() == 2 and target.size(1) in (1, F):
                if target.size(1) == 1:  # [B,1]
                    tgt_vid = target[:, 0].long()
                    tgt_frame = tgt_vid.view(B, 1).repeat(1, F)
                else:  # [B,F]
                    tgt_frame = target[:, :F].long()
                    tgt_vid = tgt_frame.float().mean(dim=1).round().long()
            else:  # 其它情况，做个稳妥降级
                t = target.view(B, -1)
                tgt_vid = t.float().mean(dim=1).round().long()
                tgt_frame = tgt_vid.view(B, 1).repeat(1, F)

            # 写入 video-level / frame-level 评分
            probs_cpu = probs.detach().cpu()
            tgt_vid_c = tgt_vid.detach().cpu()
            tgt_fr_c = tgt_frame.detach().cpu()

            for b in range(B):
                # video-level: 按帧均值
                pred_b = probs_cpu[b].mean().item()
                score_list.append(f"{pred_b} {int(tgt_vid_c[b].item())}\n")
                # frame-level
                for f in range(F):
                    score_frame_list.append(f"{probs_cpu[b, f].item()} {int(tgt_fr_c[b, f].item())}\n")

        # 写文件并沿用你现有评测流程
        map_score_filename = f"{name}_score.txt"
        with open(map_score_filename, "w") as file:
            file.writelines(score_list)
        map_score_frame_filename = f"{name}_frame_score.txt"
        with open(map_score_frame_filename, "w") as file:
            file.writelines(score_frame_list)

        test_ACC, fpr, FRR, HTER, auc_test, test_err = performances_val(map_score_filename)
        test_ACC_frame, _, _, HTER_frame, auc_test_frame, _ = performances_val(map_score_frame_filename)

        logger.info(f' * # Video-level HTER {HTER:.4f}  AUC {auc_test:.4f}  ACC {test_ACC:.4f}')
        logger.info(f' * # Frame-level HTER {HTER_frame:.4f}  AUC {auc_test_frame:.4f}  ACC {test_ACC_frame:.4f}')
        return HTER, auc_test, test_ACC


if __name__ == '__main__':
    _, config = parse_option()
    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0 / 2
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0 / 2
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0 / 2
    # # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # config.defrost()
    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    main(config)
