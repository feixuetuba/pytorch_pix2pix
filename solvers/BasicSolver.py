import logging
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dataLoaders import get_dataloader
from utils import get_cls, show_remain


class BasicSolver:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        model = get_cls("models", self.cfg['model']['name'])(self.cfg)
        dataloader = get_dataloader(self.cfg, self.cfg['stage'])


        log_dir = os.path.join(self.cfg['checkpoint_dir'], 'log')
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_dir)

        which_epoch = self.cfg.get('which_epoch', 0)
        if which_epoch != 0:
            model.load(self.cfg['checkpoint_dir'], self.cfg['nn']['name'], which_epoch)
        step = 0
        logging.info("===== Training... =====")

        epochs = self.cfg['epochs'] + 1
        for epoch in range(which_epoch+1, epochs):
            start = time.time()
            for data in dataloader:
                model.set_input(data)
                model.optimize_parameters()
                writer.add_scalars('Loss', model.get_current_error(), step)
                step += 1
            elapse = time.time() - start
            remain = (epochs - epoch) * elapse
            writer.add_image(self.cfg['name'],model.get_current_visual(3), epoch, dataformats='HWC')
            old_lr, lr = model.update_learning_rate()
            logging.info(f"[{epoch}], {old_lr} -> {old_lr}, elapse:{elapse}, remain:{show_remain(remain)}")
            epoch += 1
            if epoch % self.cfg['solver']['save_epoch'] == 0:
                model.save(self.cfg['checkpoint_dir'], 'p2p', epoch)
        logging.info("===== finished... =====")

    def test(self):
        from tqdm import tqdm
        import cv2
        model = get_cls("models", self.cfg['model']['name'])(self.cfg)
        which_epoch = self.cfg.get('which_epoch', 0)
        model.load(self.cfg['checkpoint_dir'], self.cfg['nn']['name'], which_epoch)
        logging.info("===== Testing... =====")
        img_root = self.cfg['dataset']['test']['dataroot']
        load_size = self.cfg['dataset']['crop_size']
        files = os.listdir(img_root)
        dest_dir = self.cfg['dataset']['test'].get('dest', None)
        if dest_dir is not None:
            os.makedirs(dest_dir, exist_ok=True)
        for f in tqdm(files, total=len(files)):
            img = cv2.imread(os.path.join(img_root, f))
            h, w = img.shape[:2]
            net_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            net_input = cv2.resize(net_input, (load_size,load_size), cv2.INTER_CUBIC)
            net_input = np.transpose(net_input, (2,0,1)).astype(np.float32)
            net_input = net_input[None, ...] / 127.5 - 1
            model.set_input({'A':torch.from_numpy(net_input)})
            with torch.no_grad():
                model.forward()
                result = model.fake_B.cpu().numpy()[0]
                result = np.transpose(result,(1,2,0))
                result = (result + 1) * 127.5
                result = np.clip(result, 0, 255).astype(np.uint8)
                result = cv2.resize(result, (w, h))
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                result = np.hstack([img, result])
                if dest_dir is not None:
                    dest_f = os.path.join(dest_dir, f)
                    cv2.imwrite(dest_f, result)
                else:
                    cv2.imshow(f, result)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27:
                        break
        logging.info("===== finished... =====")