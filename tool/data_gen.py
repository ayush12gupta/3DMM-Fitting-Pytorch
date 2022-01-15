import cv2, os
from facenet_pytorch import MTCNN
from ..core.options import ImageFittingOptions
from ..core import get_recon_model
import face_alignment
from ..core import utils
import torch

def generate(args, outdir, inpdir, img_fn):
    img_arr = cv2.imread(inpdir + img_fn)[:, :, ::-1]
    orig_h, orig_w = img_arr.shape[:2]
    bboxes, probs = mtcnn.detect(img_arr)
    bbox = utils.pad_bbox(bboxes[0], (orig_w, orig_h), args.padding_ratio)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    assert face_w == face_h
    face_img = img_arr[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))
    cv2.imwrite(outdir + img_fn[:-4] + ".png", resized_face_img[:, :, ::-1])
    lms = fa.get_landmarks_from_image(resized_face_img)[0]
    kp_idx = recon_model.kp_idx[0]
    lms = lms[kp_idx, :2]
    with open(outdir + img_fn[:-4] + ".txt", "w") as txt_file:
        for line in lms:
            line = [str(l) for l in line]
            txt_file.write(" ".join(line) + "\n")


if __name__ == '__main__':
    device = torch.device('cuda', 0)
    mtcnn = MTCNN(device=device, select_largest=False)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, flip_input=False)
    args = ImageFittingOptions()
    args = args.parse()
    recon_model = get_recon_model(model=args.recon_model,
                                  device=device,
                                  batch_size=1,
                                  img_size=args.tar_size)
    imdirs = os.listdir("./data")
    outdir =  "./test/"
    os.makedirs(outdir, exist_ok=True)
    for d in imdirs[:]:
        print(d)
        generate(args, outdir, "./data/", d)
