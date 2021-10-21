import os
import torch
from PIL import Image
import numpy as np
# def denormalize(image):
#     std_ = torch.tensor([[57.375, 57.12, 58.395]]).to(image.device)
#     mean_ = torch.tensor([[103.53, 116.28, 123.675]]).to(image.device)
#     image.mul_(std_).add_(mean_)
#     return image
# def vis_pss_map(img, pred_pss, ori_h, ori_w):
#     im = (norm(img.data.cpu()).numpy()*255).astype(np.uint8).transpose((1, 2, 0))
#     img = Image.fromarray(im).convert('RGB').resize((ori_w, ori_h))
#     pss = pred_pss.data.cpu().numpy()
#     pss_img = Image.fromarray((pss[0, :, :]*255).astype(np.uint8)).convert('RGB').resize((ori_w, ori_h))
#     pss_img = Image.blend(pss_img, img, 0.5)
#     return img, pss_img
# def visualize_input_images():
#     return images
# _C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# # Values to be used for image normalization
# _C.INPUT.PIXEL_STD = [1., 1., 1.]
class Visualizater(object):
    def __init__(self,):
        print("nothing")
    def _denormalize(self, image):
        std_ = torch.tensor([[57.375, 57.12, 58.395]]).to(image.device)
        mean_ = torch.tensor([[103.53, 116.28, 123.675]]).to(image.device)
        # std_ = torch.tensor([[1., 1., 1.]]).to(image.device)
        # mean_ = torch.tensor([[102.9801, 115.9465, 122.7717]]).to(image.device)
        image.mul_(std_).add_(mean_)
        return image
    def _vis_multi_image(self, image_list, shape=[1,-1]):
        # print(image_list)
        image_num = len(image_list)
        h, w,_ = np.array(image_list[0]).shape
        h = int(h)
        w = int(w)
        #print h,w
        num_w = int(image_num//shape[0])
        num_h = shape[0]
        new_im = Image.new('RGB', (num_w*w,num_h*h))
        for idx, image in enumerate(image_list):
            idx_w = int(idx%num_w)
            idx_h = int(idx/num_w)
            new_im.paste(image, (idx_w*w,idx_h*h))
        return new_im
    def conver_images_to_pil(self, images):
        images_to_show = []
        image_tensor = images.permute(0,2,3,1).float()
        image_de = self._denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)
        for image in image_de:
            # print(image.max())
            images_to_show.append(Image.fromarray(image[:,:,(2,1,0)]).convert('RGB'))
        return images_to_show
    def convert_masks_to_pil(self, masks):
        # print(masks.shape)
        # print(masks)
        masks_to_show = [] 
        for mask in masks:
            # print(mask.shape)
            mask = mask.data.cpu().numpy()
            mask = Image.fromarray((mask[0, :, :]*255).astype(np.uint8)).convert('RGB')
            masks_to_show.append(mask)
        return masks_to_show
    def render_masks_to_images(self, images, masks):
        shows = []
        for image, mask in zip(images, masks):
            w, h = image.size
            mask = mask.resize((w,h))
            shows.append(Image.blend(image, mask, 0.5))
        return shows
    def cat_images(self, image_lists, shape=[1,-1]):
        images = []
        for tuple_ in image_lists:
            image = self._vis_multi_image(list(tuple_), shape=shape)
            images.append(image)
        return images
    def save(self, images, folder, names=None):
        if names==None:
            for image in images:
                path = os.path.join(folder, "image_{}.jpg".format(np.random.randint(0,500)))
                image.save(path)
        else:
            for image, name in zip(images,names):
                path = os.path.join(folder, name)
                image.save(path)


