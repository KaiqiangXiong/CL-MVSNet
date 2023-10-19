import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .module import *

Align_Corners_Range = False

class FPNFeature(nn.Module):
    def __init__(self, args, **kwargs):
        super(FPNFeature, self).__init__()

        base_channels = args.base_channels
        self.num_stage = args.num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]
        final_chs = base_channels * 4
        if self.num_stage == 3:
            self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
            self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

            self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
            self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
            self.out_channels.append(base_channels * 2)
            self.out_channels.append(base_channels)

        elif self.num_stage == 2:
            self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

            self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
            self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}

        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.num_stage == 3:
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
            out = self.out2(intra_feat)
            outputs["stage2"] = out


            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
            out = self.out3(intra_feat)
            outputs["stage3"] = out

        elif self.num_stage == 2:
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
            out = self.out2(intra_feat)
            outputs["stage2"] = out

        return outputs

class InitSampler(nn.Module):
    def __init__(self, sample, **kwargs):
        super(InitSampler, self).__init__()
        self.num_hypotheses = sample['num_hypotheses']

    def forward(self, last_depth, shape, interval_base=None, **kwargs):

        assert last_depth.dim() == 2, "Initial reference depth must be a range with dim 2!!"

        last_depth_min = last_depth[:, 0]  # (B,)
        last_depth_max = last_depth[:, -1]
        new_interval = (last_depth_max - last_depth_min) / (self.num_hypotheses - 1)  # (B, )

        depth_hypotheses = last_depth_min.unsqueeze(1) + (
                torch.arange(0, self.num_hypotheses, device=last_depth.device, dtype=last_depth.dtype,
                             requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1))  # (B, D)

        # (B, D, H, W)
        depth_hypotheses = depth_hypotheses.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])

        return depth_hypotheses

class UniformSampler(nn.Module):
    def __init__(self, sample, **kwargs):
        super(UniformSampler, self).__init__()
        self.num_hypotheses = sample['num_hypotheses']
        self.interval_ratio = sample['interval_ratio']

    def forward(self, last_outs, shape, interval_base=None, **kwargs):
        last_depth = last_outs["depth"].detach()

        depth_interval = self.interval_ratio * interval_base

        last_depth_min = (last_depth - self.num_hypotheses / 2 * depth_interval)  # (B, H, W)
        last_depth_max = (last_depth + self.num_hypotheses / 2 * depth_interval)

        new_interval = (last_depth_max - last_depth_min) / (self.num_hypotheses - 1)  # (B, H, W)

        depth_hypotheses = last_depth_min.unsqueeze(1) + (torch.arange(0, self.num_hypotheses, device=last_depth.device,
                                                                       dtype=last_depth.dtype, requires_grad=False).reshape(1, -1, 1, 1)
                                                          * new_interval.unsqueeze(1))

        depth_hypotheses = F.interpolate(depth_hypotheses, shape, mode='bilinear', align_corners=False)

        return depth_hypotheses
    
class GroupWiseAgg(nn.Module):
    def __init__(self, args, **kwargs):
        super(GroupWiseAgg, self).__init__()
        self.G = args.group

        self.out_channels = self.G

    def forward(self, features, proj_matrices, depth_hypotheses, **kwargs):
        """
        :param features: [ref_fea, src_fea1, src_fea2, ...], fea shape: (b, c, h, w)
        :param proj_matrices: (b, nview, ...) [ref_proj, src_proj1, src_proj2, ...]
        :param depth_hypotheses: (b, ndepth, h, w)
        :return: matching cost volume (b, c, ndepth, h, w)
        """
        ref_feature, src_features = features[0], features[1:]
        proj_matrices = torch.unbind(proj_matrices, 1)  # to list
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        num_views = len(features)
        num_depth = depth_hypotheses.shape[1]

        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        b, c, d, h, w = ref_volume.shape
        ref_volume = ref_volume.view(b, self.G, c // self.G, d, h, w)
        volume_sum = None

        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypotheses)

            warped_volume = warped_volume.view_as(ref_volume)
            if volume_sum is None:
                volume_sum = warped_volume
            else:
                volume_sum = volume_sum + warped_volume

            del warped_volume

        volume_correlation  = torch.mean(volume_sum * ref_volume, dim=2) / (num_views - 1)  # (b, g, d, h, w)

        # cost_volume = {"volume_correlation":volume_correlation}
        return volume_correlation

class UNet3DCNNReg(nn.Module):
    def __init__(self, args, in_channels=8, **kwargs):
        super(UNet3DCNNReg, self).__init__()

        base_channels = args.base_channels

        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, **kwargs):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class RegressionDepth(nn.Module):
    def __init__(self, args, **kwargs):
        super(RegressionDepth, self).__init__()

    def forward(self, cost_reg, depth_hypotheses, **kwargs):

        prob_volume_pre = cost_reg.squeeze(1)  # (b, d, h, w)

        prob_volume = F.softmax(prob_volume_pre, dim=1)  # (b, ndepth, h, w)
        depth = depth_regression(prob_volume, depth_hypotheses=depth_hypotheses)  # (b, h, w)

        num_depth = prob_volume.shape[1]

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1,
                                                padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume,
                                           depth_hypotheses=torch.arange(num_depth, device=prob_volume.device,
                                                                         dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            pv = torch.where(prob_volume <= 0, torch.ones_like(prob_volume)*1e-5, prob_volume)
            distribution_consistency = (np.log(pv.shape[1]) - torch.sum(-pv * torch.log(pv), dim=1)) / np.log(pv.shape[1])
            # photometric_confidence[distribute_quality > 0.9] = 0

        return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume,
                "depth_hypotheses": depth_hypotheses, "depth_mode": "regression",
                "distribution_consistency": distribution_consistency}

class CasMVSNet(nn.Module):
    def __init__(self, args):
        super(CasMVSNet, self).__init__()
        self.feature = FPNFeature(args)
        self.sampler = nn.ModuleList([InitSampler(args.sample1), UniformSampler(args.sample2), UniformSampler(args.sample3)])
        self.aggregation = nn.ModuleList([GroupWiseAgg(args), GroupWiseAgg(args), GroupWiseAgg(args)])
        self.regularization = nn.ModuleList([UNet3DCNNReg(args), UNet3DCNNReg(args), UNet3DCNNReg(args)])
        self.depth_head = nn.ModuleList([RegressionDepth(args), RegressionDepth(args), RegressionDepth(args)])
        self.num_stage = args.num_stage
        self.args = args

    def forward(self, data, icc=False, scc=False, epoch=0):
        outputs = {} 
        imgs = data["imgs"]               
        proj_matrices = data["proj_matrices"] 
        
        if icc:
            imgs_icc = data["imgs_aug"] # b v c h w
            nviews = imgs_icc.shape[1]
            for view_idx in range(1, nviews, 1):   # loops through view dimension instead of batch dimension
                per = min(self.args.p_icc * epoch / 15, self.args.p_icc)
                mask = torch.ones_like(imgs_icc[:, view_idx]) * per
                mask = 1 - mask.bernoulli() 
                imgs_icc[:, view_idx] = imgs_icc[:, view_idx] * mask
            ref_img = imgs_icc[:, 0]
            ref_img, filter_mask = random_image_mask(ref_img, filter_size=(ref_img.size(2) // 3, ref_img.size(3) // 3))
            imgs_icc[:, 0] = ref_img
            imgs = imgs_icc
            outputs["filter_mask"] = filter_mask
        if scc:
            imgs = data["imgs_scc"]
            proj_matrices = data["proj_matrices_scc"]

        init_depth_hypotheses  = data["init_depth_hypotheses"]
        interval_base = (init_depth_hypotheses[0, -1] - init_depth_hypotheses[0, 0]) / init_depth_hypotheses.size(1)
        
        features = []
        for nview_idx in range(imgs.size(1)):  
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        
        for stage_idx in range(self.num_stage):
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_shape = features_stage[0].shape[2:]

            if stage_idx == 0:
                last_outs = init_depth_hypotheses
            else:
                last_outs = outputs["stage{}".format(stage_idx)]

            depth_hypotheses = self.sampler[stage_idx](last_outs, stage_shape, interval_base)
            cost_volume = self.aggregation[stage_idx](features_stage, proj_matrices_stage, depth_hypotheses)
            cost_reg = self.regularization[stage_idx](cost_volume)
            # depth
            outputs_stage = self.depth_head[stage_idx](cost_reg=cost_reg, depth_hypotheses=depth_hypotheses)
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs

class CLMVSNet(nn.Module):
    def __init__(self, args):
        super(CLMVSNet, self).__init__()
        self.model = CasMVSNet(args)

    def forward(self, data, mode, epoch=0):
        assert mode in ["train", "val", "test"], "mode wrong!"
        outputs = {}
        output1 = self.model(data)
        outputs["output1"] = output1
        if mode in ["train", "val"]:
            output2 = self.model(data, icc=True, epoch=epoch)
            output3 = self.model(data, scc=True)
            outputs["output2"] = output2     
            outputs["output3"] = output3
        outputs.update(output1)
        return outputs
