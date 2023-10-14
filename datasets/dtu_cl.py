import torch
from torch.utils.data import Dataset
import numpy as np
import os, cv2
from PIL import Image
from torchvision import transforms
from datasets.data_io import *

class MVSDataset(Dataset):
    def __init__(self, args, list_file, mode):
        super(MVSDataset, self).__init__()
        self.img_size = args.img_size
        self.datapath = args.datapath
        self.listfile = list_file
        self.mode = mode
        self.nviews = args.nviews         
        self.ndepths = args.numdepth
        self.interval_scale = args.interval_scale
        self.random_view = False

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.define_transforms()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        self.id_list = []
        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]       
                    if self.mode == "val":
                        for light_idx in range(7):
                            metas.append((scan, light_idx, ref_view, src_views))
                            self.id_list.append([ref_view] + src_views)
                    else:
                        assert self.mode == "train"
                        for light_idx in range(7):
                            metas.append((scan, light_idx, ref_view, src_views))
                            self.id_list.append([ref_view] + src_views)
        self.id_list = np.unique(self.id_list)
        self.build_remap()
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i    

    def define_transforms(self):
        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            RandomGamma(min_gamma=0.5, max_gamma=2.0, clip_image=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_seg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) 
        depth_interval = float(lines[11].split()[1]) * self.interval_scale 
        depth_max = depth_min + depth_interval * self.ndepths
        return intrinsics, extrinsics, depth_min, depth_interval, [depth_min, depth_max]

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.0
        return np_img

    def read_img_seg(self, filename):
        img = Image.open(filename)
        return self.transform_seg(img)


    def read_img_aug(self, filename):
        img = Image.open(filename)
        img = self.transform_aug(img)
        return img

    def center_image(self, img):
        """ normalize image input """
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def prepare_img(self, hr_img):
        # downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        # crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        return hr_img_crop

    def read_mask_hr_crop(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = np_img[:1184,:]
        np_img = (np_img > 10).astype(np.float32)
        # np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms   

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms   
    
    def read_depth_all(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                                interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=1.0, fy=1.0,
                                interpolation=cv2.INTER_NEAREST) 

        return depth_h

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)


    def read_depth_hr(self, filename):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)
        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms
    
    def read_depth_hr_crop(self, filename):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = depth_hr[:1184,:]
        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        sample = {}
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if not self.random_view:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
        else:
            num_src_views = len(src_views)
            rand_ids = torch.randperm(num_src_views)[:self.nviews - 1]      
            src_views_t = torch.tensor(src_views)                          
            view_ids = [ref_view] + list(src_views_t[rand_ids].numpy())     
        num_src_views = len(src_views)
        rand_ids = torch.randperm(num_src_views)[:self.nviews - 1]      
        src_views_t = torch.tensor(src_views)                         
        view_ids_scc = [ref_view] + list(src_views_t[rand_ids].numpy())
        imgs = []
        imgs_aug = []
        center_imgs = []
        mask = None
        depth_values = None
        init_depth_hypotheses = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            # img = self.read_img(img_filename)
            image_aug = self.read_img_aug(img_filename)
            image_seg = self.read_img_seg(img_filename)
            center_img = self.center_image(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB))           
            intrinsics, extrinsics, depth_min, depth_interval, _ = self.read_cam_file(proj_mat_filename)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            if i == 0:  # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)
                # get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                init_depth_hypotheses = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                mask = mask_read_ms
            imgs.append(image_seg)
            imgs_aug.append(image_aug)
            center_imgs.append(center_img)
        imgs = np.stack(imgs)
        center_imgs = np.stack(center_imgs).transpose([0, 3, 1, 2])
        imgs_aug = torch.stack(imgs_aug)

        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }
        sample["imgs"] = imgs
        sample["imgs_aug"] = imgs_aug
        sample["proj_matrices"] = proj_matrices_ms
        sample["depth"] = depth_ms
        sample["depth_values"] = depth_values
        sample["mask"] = mask
        sample["center_imgs"] = center_imgs
        sample["view_ids"] = view_ids
        sample["view_ids_scc"] = view_ids_scc
        sample['init_depth_hypotheses'] = init_depth_hypotheses.astype(np.float32)
        imgs_scc = []
        proj_matrices_scc = []

        for i, vid in enumerate(view_ids_scc):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            image_scc = self.read_img_seg(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval, _ = self.read_cam_file(proj_mat_filename)
            proj_mat_scc = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_scc[0, :4, :4] = extrinsics
            proj_mat_scc[1, :3, :3] = intrinsics
            proj_matrices_scc.append(proj_mat_scc)
            imgs_scc.append(image_scc)
        imgs_scc = np.stack(imgs_scc)
        proj_matrices_scc = np.stack(proj_matrices_scc)
        stage2_pjmats_scc = proj_matrices_scc.copy()
        stage2_pjmats_scc[:, 1, :2, :] = proj_matrices_scc[:, 1, :2, :] * 2
        stage3_pjmats_scc = proj_matrices_scc.copy()
        stage3_pjmats_scc[:, 1, :2, :] = proj_matrices_scc[:, 1, :2, :] * 4
        proj_matrices_ms_scc = {
            "stage1": proj_matrices_scc,
            "stage2": stage2_pjmats_scc,
            "stage3": stage3_pjmats_scc
        }
        sample["imgs_scc"] = imgs_scc
        sample["proj_matrices_scc"] = proj_matrices_ms_scc

        return sample

