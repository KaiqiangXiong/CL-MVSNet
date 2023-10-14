import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class MVSLoss(nn.Module):
    def __init__(self, args):
        super(MVSLoss, self).__init__()
        self.loss_funcs = [UnsupLossMultiStage_l05(args), ICCLossMultiStage(args), SCCLossMultiStage(args)]    
        self.args = args

    def forward(self, data, outputs, epoch_idx):
        losses = {}
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=data["imgs"].device, requires_grad=False)
        for loss_func in self.loss_funcs:
            loss, _ = loss_func(data, outputs, epoch_idx)
            losses[loss_func.name] = loss.item()
            total_loss = total_loss + loss
        return total_loss, losses

class UnSupLoss(nn.Module):
    def __init__(self,args):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()
        self.args = args

    def forward(self, imgs, cams, depth, stage_idx):
        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        num_views = len(imgs)
        ref_img = imgs[0]
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0
        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)  
            if mask.sum() == 0:
                self.unsup_loss = torch.tensor(0.0, dtype=torch.float32, device=mask.device)
                return self.unsup_loss
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss_l0_5(warped_img, ref_img, mask, simple=False)  
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)
            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))  
        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)   
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))   
        self.unsup_loss = self.args.wrecon * self.reconstr_loss + 6 * self.ssim_loss + 0.18 * self.smooth_loss
        return self.unsup_loss

class UnsupLossMultiStage_l05(nn.Module):
    def __init__(self, args):
        super(UnsupLossMultiStage_l05, self).__init__()
        self.name = "unslossl05"
        self.args = args
        self.unsup_loss = UnSupLoss(args)

    def forward(self, data, outputs, epoch_idx, **kwargs):
        inputs = outputs
        imgs = data["center_imgs"]
        cams = data["proj_matrices"]
        depth_loss_weights = self.args.dlossw
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)
        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)
            if depth_loss_weights is not None:
                total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss = total_loss + 1.0 * depth_loss
            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss
        return total_loss, scalar_outputs

class ICCLossMultiStage(nn.Module):
    def __init__(self, args):
        super(ICCLossMultiStage, self).__init__()
        self.name = "iccloss"
        self.args = args

    # def forward(self, inputs, pseudo_depth, mask_ms, filter_mask, **kwargs):
    def forward(self, data, outputs, epoch_idx, **kwargs):
        if not "output2" in outputs: return torch.tensor(0.0, dtype=torch.float32, device=data["imgs"].device, requires_grad=False), {}
        inputs = outputs["output2"]
        pseudo_depth = outputs["output1"]["depth"].detach()
        filter_mask = inputs["filter_mask"] 
        depth_loss_weights = self.args.dlossw
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=pseudo_depth.device, requires_grad=False)
        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]

            pseudo_gt = pseudo_depth.unsqueeze(dim=1)
            if stage_idx == 0:
                pseudo_gt_t = F.interpolate(pseudo_gt, scale_factor=(0.25, 0.25))
                filter_mask_t = F.interpolate(filter_mask, scale_factor=(0.25, 0.25))
            elif stage_idx == 1:
                pseudo_gt_t = F.interpolate(pseudo_gt, scale_factor=(0.5, 0.5))
                filter_mask_t = F.interpolate(filter_mask, scale_factor=(0.5, 0.5))
            else:
                pseudo_gt_t = pseudo_gt
                filter_mask_t = filter_mask
            filter_mask_t = filter_mask_t[:, 0, :, :]
            pseudo_gt_t = pseudo_gt_t.squeeze(dim=1)
            mask = filter_mask_t > 0.5
            depth_loss = F.smooth_l1_loss(depth_est[mask], pseudo_gt_t[mask], reduction='mean')
            if depth_loss_weights is not None:
                total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss = total_loss + 1.0 * depth_loss
            scalar_outputs["mva_loss_stage{}".format(stage_idx + 1)] = depth_loss
        w_icc = adjust_w_icc(epoch_idx, self.args.w_icc, self.args.max_w_icc)
        total_loss = total_loss * w_icc

        return total_loss, scalar_outputs

class SCCLossMultiStage(nn.Module):
    def __init__(self, args, **kwargs):
        super(SCCLossMultiStage, self).__init__()
        self.name = "sccloss"
        self.conf = args.mask_conf
        self.args = args

    def forward(self, data, outputs, epoch_idx, **kwargs):
        depth_loss_weights = self.args.dlossw
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=data["center_imgs"].device, requires_grad=False)
        scalar_outputs = {}
        photometric_confidence = outputs["output1"]["photometric_confidence"].clone().detach()
        pseudo_depth = outputs["output1"]["depth"].clone().detach()
        output3 = outputs["output3"]
        w_scc = self.args.w_scc
        for stage_key in [k for k in output3.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1     # 0 1 2
            pseudo_gt = pseudo_depth.unsqueeze(dim=1)
            photometric_confidence_tp = photometric_confidence.unsqueeze(dim=1)
            if stage_idx == 0:
                pseudo_gt_t = F.interpolate(pseudo_gt, scale_factor=(0.25, 0.25))
                photometric_confidence_t = F.interpolate(photometric_confidence_tp, scale_factor=(0.25, 0.25))
                mask_t = photometric_confidence_t > self.conf
            elif stage_idx == 1:
                pseudo_gt_t = F.interpolate(pseudo_gt, scale_factor=(0.5, 0.5))
                photometric_confidence_t = F.interpolate(photometric_confidence_tp, scale_factor=(0.5, 0.5))
                mask_t = photometric_confidence_t > self.conf
            else:
                pseudo_gt_t = pseudo_gt
                photometric_confidence_t = photometric_confidence_tp
                mask_t = photometric_confidence_t > self.conf
            pseudo_gt_t = pseudo_gt_t.squeeze(dim=1)
            mask_t = mask_t.squeeze(dim=1)
            if torch.sum(mask_t.type(torch.float32)) == 0:
                depth_loss = torch.tensor(0.0, dtype=torch.float32, device=data["center_imgs"].device)
            else:
                depth_loss = F.smooth_l1_loss(pseudo_gt_t[mask_t], output3[stage_key]["depth"][mask_t], reduction='mean')
            if depth_loss_weights is not None:
                total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss = total_loss + 1.0 * depth_loss
        total_loss = total_loss * w_scc
        return total_loss, scalar_outputs