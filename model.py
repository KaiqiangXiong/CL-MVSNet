import cv2
import time
import progressbar
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from networks.clmvsnet import CLMVSNet
from datasets import get_loader
from tools import *
from loss import MVSLoss
from datasets.data_io import save_pfm
from filter import pcd_filter
from torchvision import transforms

class Model:
    def __init__(self, args):

        cudnn.benchmark = True
        init_distributed_mode(args)

        self.args = args
        self.device = torch.device("cpu" if self.args.no_cuda or not torch.cuda.is_available() else "cuda")
        self.network = CLMVSNet(args).to(self.device)

        if self.args.distributed and self.args.sync_bn:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)

        if not (self.args.val or self.args.test):
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=args.lr,
                                              weight_decay=args.wd)
            self.lr_scheduler = get_schedular(self.optimizer, self.args)
            self.train_loader, self.train_sampler = get_loader(args, args.trainlist, "train")

        if not self.args.test:
            self.loss_func = MVSLoss(args)
            self.val_loader, self.val_sampler = get_loader(args, args.testlist, "val")
            if is_main_process():
                self.writer = SummaryWriter(log_dir=args.log_dir, comment="Record network info")

        self.network_without_ddp = self.network
        if self.args.distributed:
            self.network = DistributedDataParallel(self.network, device_ids=[self.args.local_rank])
            self.network_without_ddp = self.network.module

        if self.args.resume:
            checkpoint = torch.load(self.args.resume, map_location="cpu")
            if not (self.args.val or self.args.test):
                self.args.start_epoch = checkpoint["epoch"] + 1
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.network_without_ddp.load_state_dict(checkpoint["model"])
            
        self.args = args

    def main(self):
        if self.args.val:
            self.validate()
            return
        if self.args.test:
            self.test()
            return
        self.train()

    def train(self):
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            if is_main_process():
                torch.save({
                    'epoch': epoch,
                    'model': self.network_without_ddp.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(self.args.log_dir, epoch))

            if (epoch % self.args.eval_freq == 0) or (epoch == self.args.epochs - 1):
                self.validate(epoch)
            torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        self.network.train()

        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                        progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('LR', width=1), ",",
                        progressbar.Variable('Loss', width=1), ",", progressbar.Variable('Th2', width=1), ",",
                        progressbar.Variable('Th4', width=1), ",", progressbar.Variable('Th8', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.train_loader),
                                           prefix="Epoch {}/{}: ".format(epoch, self.args.epochs)).start()

        avg_scalars = DictAverageMeter()

        for batch, data in enumerate(self.train_loader):
            data = tocuda(data)

            outputs = self.network(data, "train", epoch)

            loss, losses= self.loss_func(data, outputs, epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step(epoch + batch / len(self.train_loader))

            gt_depth = data["depth"]["stage{}".format(self.args.num_stage)]
            mask = data["mask"]["stage{}".format(self.args.num_stage)]
            thres2mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 2)
            thres4mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 4)
            thres8mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 8)
            abs_depth_error = AbsDepthError_metrics(outputs["depth"], gt_depth, mask > 0.5)

            scalar_outputs = {"loss": loss,
                              "abs_depth_error": abs_depth_error,
                              "thres2mm_error": thres2mm,
                              "thres4mm_error": thres4mm,
                              "thres8mm_error": thres8mm}

            image_outputs = {"depth_est": outputs["depth"] * mask,
                             "depth_est_nomask": outputs["depth"],
                             "depth_gt": gt_depth,
                             "ref_img": data["imgs"][:, 0],
                             "mask": mask,
                             "errormap": (outputs["depth"] - gt_depth).abs() * mask,
                             }

            if self.args.distributed:
                scalar_outputs = reduce_scalar_outputs(scalar_outputs)

            scalar_outputs, image_outputs = tensor2float(scalar_outputs), tensor2numpy(image_outputs)

            if is_main_process():
                avg_scalars.update(scalar_outputs)
                if batch >= len(self.train_loader) - 1:
                    save_scalars(self.writer, 'train_avg', avg_scalars.avg_data, epoch)
                if (epoch * len(self.train_loader) + batch) % self.args.summary_freq == 0:
                    save_scalars(self.writer, 'train', scalar_outputs, epoch * len(self.train_loader) + batch)
                    save_images(self.writer, 'train', image_outputs, epoch * len(self.train_loader) + batch)

                pbar.update(batch, LR=self.optimizer.param_groups[0]['lr'],
                            Loss="{:.3f}|{:.3f}".format(scalar_outputs["loss"], avg_scalars.avg_data["loss"]),
                            Th2="{:.3f}|{:.3f}".format(scalar_outputs["thres2mm_error"], avg_scalars.avg_data["thres2mm_error"]),
                            Th4="{:.3f}|{:.3f}".format(scalar_outputs["thres4mm_error"], avg_scalars.avg_data["thres4mm_error"]),
                            Th8="{:.3f}|{:.3f}".format(scalar_outputs["thres8mm_error"], avg_scalars.avg_data["thres8mm_error"]))

        if is_main_process():
            pbar.finish()

    @torch.no_grad()
    def validate(self, epoch=0):
        self.network.eval()

        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                        progressbar.Timer(), ",", progressbar.ETA(), ",", progressbar.Variable('Loss', width=1), ",",
                        progressbar.Variable('Th2', width=1), ",", progressbar.Variable('Th4', width=1), ",",
                        progressbar.Variable('Th8', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.val_loader), prefix="Val:").start()

        avg_scalars = DictAverageMeter()

        for batch, data in enumerate(self.val_loader):
            data = tocuda(data)

            outputs = self.network(data,"val")
            
            loss, losses = self.loss_func(data, outputs, epoch)

            gt_depth = data["depth"]["stage{}".format(self.args.num_stage)]
            mask = data["mask"]["stage{}".format(self.args.num_stage)]
            thres2mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 2)
            thres4mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 4)
            thres8mm = Thres_metrics(outputs["depth"], gt_depth, mask > 0.5, 8)
            abs_depth_error = AbsDepthError_metrics(outputs["depth"], gt_depth, mask > 0.5)

            scalar_outputs = {"loss": loss,
                              "abs_depth_error": abs_depth_error,
                              "thres2mm_error": thres2mm,
                              "thres4mm_error": thres4mm,
                              "thres8mm_error": thres8mm}

            image_outputs = {"depth_est": outputs["depth"] * mask,
                             "depth_est_nomask": outputs["depth"],
                             "depth_gt": gt_depth,
                             "ref_img": data["imgs"][:, 0],
                             "mask": mask,
                             "errormap": (outputs["depth"] - gt_depth).abs() * mask,
                             }

            if self.args.distributed:
                scalar_outputs = reduce_scalar_outputs(scalar_outputs)

            scalar_outputs, image_outputs = tensor2float(scalar_outputs), tensor2numpy(image_outputs)

            if is_main_process():
                avg_scalars.update(scalar_outputs)
                if batch >= len(self.val_loader) - 1:
                    save_scalars(self.writer, 'test_avg', avg_scalars.avg_data, epoch)
                if (epoch * len(self.val_loader) + batch) % self.args.summary_freq == 0:
                    save_scalars(self.writer, 'test', scalar_outputs, epoch * len(self.val_loader) + batch)
                    save_images(self.writer, 'test', image_outputs, epoch * len(self.val_loader) + batch)

                pbar.update(batch,
                            Loss="{:.3f}|{:.3f}".format(scalar_outputs["loss"], avg_scalars.avg_data["loss"]),
                            Th2="{:.3f}|{:.3f}".format(scalar_outputs["thres2mm_error"], avg_scalars.avg_data["thres2mm_error"]),
                            Th4="{:.3f}|{:.3f}".format(scalar_outputs["thres4mm_error"], avg_scalars.avg_data["thres4mm_error"]),
                            Th8="{:.3f}|{:.3f}".format(scalar_outputs["thres8mm_error"], avg_scalars.avg_data["thres8mm_error"]))

        if is_main_process():
            pbar.finish()

    @torch.no_grad()
    def test(self):
        inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
        )
        self.network.eval()

        with open(self.args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
            
        num_stage = self.args.num_stage

        # step1. save all the depth maps and the masks in outputs directory
        for scene in testlist:

            TestImgLoader, _ = get_loader(self.args, [scene], "test")

            for batch_idx, data in enumerate(TestImgLoader):
                data_cuda = tocuda(data)
                start_time = time.time()
                outputs = self.network(data_cuda,"test")
                end_time = time.time()
                outputs = tensor2numpy_str(outputs)
                del data_cuda
                filenames = data["filename"]
                cams = data["proj_matrices"]["stage{}".format(num_stage)].numpy()
                imgs = data["imgs"]
                print(scene,'Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

                # save depth maps and confidence maps
                for filename, cam, img, depth_est, photometric_confidence, photometric_confidence2, photometric_confidence1 \
                        in zip(filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"],
                        outputs["stage2"]["photometric_confidence"], outputs["stage1"]["photometric_confidence"]):

                    h, w = photometric_confidence.shape
                    img = img[0]  
                    img = inv_normalize(img).numpy()     
                    cam = cam[0]  
                    photometric_confidence2 = cv2.resize(photometric_confidence2, (w, h), interpolation=cv2.INTER_NEAREST)
                    photometric_confidence1 = cv2.resize(photometric_confidence1, (w, h), interpolation=cv2.INTER_NEAREST)
                    confidence_filename2 = os.path.join(self.args.outdir, filename.format('confidence', '_stage2.pfm'))
                    confidence_filename1 = os.path.join(self.args.outdir, filename.format('confidence', '_stage1.pfm'))
                    confidence_filename = os.path.join(self.args.outdir, filename.format('confidence', '.pfm'))
                    depth_filename = os.path.join(self.args.outdir, filename.format('depth_est', '.pfm'))
                    cam_filename = os.path.join(self.args.outdir, filename.format('cams', '_cam.txt'))
                    img_filename = os.path.join(self.args.outdir, filename.format('images', '.png'))
                    os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                    # save depth maps
                    save_pfm(depth_filename, depth_est)
 
                    # save confidence maps
                    save_pfm(confidence_filename, photometric_confidence)
                    save_pfm(confidence_filename2, photometric_confidence2)
                    save_pfm(confidence_filename1, photometric_confidence1)
                    # save cams, img
                    write_cam(cam_filename, cam)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)
                    
            torch.cuda.empty_cache()

        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        if self.args.filter_method == "pcd":
            pcd_filter(self.args, testlist, self.args.num_worker)