from facenet_pytorch import MTCNN
from core.options import ImageFittingOptions
import cv2
import face_alignment
import numpy as np
from core import get_recon_model
import os
import torch
import torch.optim as optim
import core.utils as utils
from core.encoder import Encoder
from core.dataset import ImageFolderDataset
from tqdm import tqdm
import core.losses as losses


def train(train_loader, model, recon_model, optimizer, device):
    kp_idx = recon_model.kp_idx[0]
    lm_weights = utils.get_lm_weights(device)
    for batch_idx, (raw_imgs, ldmks) in enumerate(train_loader):
        raw_imgs = (raw_imgs.to(device).to(torch.float32) / 127.5 - 1)
        ldmks = ldmks.to(device).to(torch.float32)

        coeffs = model(raw_imgs)
        pred_dict = recon_model(coeffs, render=True)
        rendered_img = pred_dict['rendered_img']
        lms_proj = pred_dict['lms_proj'][:,kp_idx,:]

        mask = rendered_img[:, :, :, 3].detach()
        raw_imgs = (raw_imgs+1)*127.5
        photo_loss_val = losses.photo_loss(
            rendered_img[:, :, :, :3], raw_imgs, mask > 0)

        lm_loss_val = losses.lm_loss(lms_proj, ldmks, lm_weights[kp_idx],
                                     img_size=args.tar_size)
        id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
        exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())
        tex_reg_loss = losses.get_l2(recon_model.get_tex_tensor())

        loss = lm_loss_val*args.lm_loss_w + \
            id_reg_loss*args.id_reg_w + \
            exp_reg_loss*args.exp_reg_w + \
            photo_loss_val*args.rgb_loss_w + \
            tex_reg_loss*args.tex_reg_w

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%50==0:
            loss_str = ''
            loss_str += 'lm_loss: %f\t' % lm_loss_val.detach().cpu().numpy()
            loss_str += 'photo_loss: %f\t' % photo_loss_val.detach().cpu().numpy()
            loss_str += 'id_reg_loss: %f\t' % id_reg_loss.detach().cpu().numpy()
            loss_str += 'exp_reg_loss: %f\t' % exp_reg_loss.detach().cpu().numpy()
            loss_str += 'tex_reg_loss: %f\t' % tex_reg_loss.detach().cpu().numpy()
            print(loss_str)

def fit(args):
    device = torch.device('cuda', 0)
    recon_model = get_recon_model(model=args.recon_model,
                                  device=args.device,
                                  batch_size=args.train_batch,
                                  img_size=args.tar_size)

    dataset = ImageFolderDataset(args.data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers)
    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), 1e-3, betas=(0, 0.99), weight_decay=0.01)
    train(train_loader, model, recon_model, optimizer, device)

# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']


if __name__ == '__main__':
    args = ImageFittingOptions()
    args = args.parse()
    args.device = 'cuda:%d' % args.gpu
    fit(args)
