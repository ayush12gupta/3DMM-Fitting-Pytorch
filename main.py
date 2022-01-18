from facenet_pytorch import MTCNN
from core.options import ImageFittingOptions
import cv2
import face_alignment
import torchvision
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
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('./log')


def train(train_loader, model, recon_model, optimizer, device):
    kp_idx = recon_model.kp_idx[0]
    lm_weights = utils.get_lm_weights(device)
    for epoch in range(100):
        lm_loss = 0.0
        photo_loss = 0.0
        exp_loss = 0.0
        id_loss = 0.0
        tex_loss = 0.0
        tloss = 0.0
        for batch_idx, (raw_imgs, ldmks) in enumerate(train_loader):
            raw_imgs = (raw_imgs.to(device).to(torch.float32) / 127.5 - 1)
            ldmks = ldmks.to(device).to(torch.float32)

            optimizer.zero_grad()

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

            lm_loss += lm_loss_val.item()
            photo_loss += photo_loss_val.item()
            exp_loss += exp_reg_loss.item()
            id_loss += id_reg_loss.item()
            tex_loss += tex_reg_loss.item()

            loss = lm_loss_val*args.lm_loss_w + \
                id_reg_loss*args.id_reg_w + \
                exp_reg_loss*args.exp_reg_w + \
                photo_loss_val*args.rgb_loss_w + \
                tex_reg_loss*args.tex_reg_w
            
            tloss += loss.item()

            loss.backward()
            optimizer.step()

            # if batch_idx%20==0:
            if batch_idx % 100 == 99:    # every 1000 mini-batches...
                writer.add_scalar('Landmark loss',
                                lm_loss / 100,
                                epoch * len(train_loader) + batch_idx)
                lm_loss = 0.0
                writer.add_scalar('ID loss',
                                id_loss / 100,
                                epoch * len(train_loader) + batch_idx)
                id_loss = 0.0
                writer.add_scalar('Expression loss',
                                exp_loss / 100,
                                epoch * len(train_loader) + batch_idx)
                exp_loss = 0.0
                writer.add_scalar('Photo loss',
                                photo_loss / 100,
                                epoch * len(train_loader) + batch_idx)
                photo_loss = 0.0
                writer.add_scalar('Texture loss',
                                tex_loss / 100,
                                epoch * len(train_loader) + batch_idx)
                tex_loss = 0.0
                writer.add_scalar('Total Loss',
                                tloss / 100,
                                epoch * len(train_loader) + batch_idx)
                tloss = 0.0
                im = np.array(rendered_img.detach().cpu())[:,:,:,::-1][:,:,:,1:]
                raw_im = np.array(raw_imgs.detach().cpu())[:,::-1,:,:]
                raw_im = np.transpose(raw_im, (0, 2, 3, 1))
                mask = mask.detach().cpu()
                ma = np.stack([mask, mask, mask], axis=3)
                raw_im[ma>0] = im[ma>0]
                raw_im = np.transpose(raw_imm, (0, 3, 1, 2))
                raw_imm = [torch.tensor(r.copy()) for r in raw_imm]
                img_grid = torchvision.utils.make_grid(raw_imm, nrow=4)
                writer.add_image('Images', img_grid)
                # cv2.imwrite("test.png", im)
                # cv2.imwrite("mask.png", ma[:,:,:]*255)
                cv2.imwrite("raw.png", raw_im[0])
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
    model = Encoder(3, args.checkpoint_dir + '/resnet50-0676ba61.pth').to(device)
    optimizer = optim.Adam(model.parameters(), 1e-4, betas=(0, 0.99), weight_decay=0.01)
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