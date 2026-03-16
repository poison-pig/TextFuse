import os
import sys
import clip
import torch
from tqdm import tqdm
from torchvision import transforms
from prompt_dataset import PromptDataSet
from model import TextFuse as create_model
from PIL import Image
import time

@torch.no_grad()
def evaluate(model, model_clip, data_loader, device):
    model.eval()
    model_clip.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)

    for  image_ir, vis_text, ir_text,  name,vis_y_image, vis_cb_image, vis_cr_image ,h ,w,flag,vis_target_text  in data_loader:

        vis_text = clip.tokenize(vis_text).to(device)
        ir_text = clip.tokenize(ir_text).to(device)
        
        vis_y_image = vis_y_image.to(device)
        image_ir = image_ir.to(device)

        vis_cb_image = vis_cb_image.to(device)
        vis_cr_image = vis_cr_image.to(device)
        vis_target_text = clip.tokenize(vis_target_text).to(device)

        I_fused = model(vis_y_image, image_ir,vis_text, ir_text, vis_target_text)

        # from thop import profile
        # flops, params = profile(model, inputs=(vis_y_image, image_ir,vis_text, ir_text, vis_target_text), verbose=True)
        # print('thop: FLOPs = ' + str(flops / 1000 ** 2) + 'M')
        # print('thop: Params = ' + str(params / 1000 ** 2) + 'M')

        fused_img = clamp(I_fused)
        fused_img_tensor = YCrCb2RGB(fused_img[0], vis_cb_image[0], vis_cr_image[0])
        fused_img = transforms.ToPILImage()(fused_img_tensor)
        
        
        if flag ==1:
            fused_img = transforms.ToPILImage()(fused_img_tensor).resize((w, h), resample=Image.BICUBIC)


        
        save_path = "./resluts/MSRS"
        if not os.path.exists(save_path):#检查目录是否存在
                os.makedirs(save_path)


        fused_img.save(os.path.join(save_path, name[0]))

    return 0




def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)



if __name__ == '__main__':

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PromptDataSet("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             drop_last=True)
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)
    model_weight_path = "./checkpoint/checkpoint.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=False)['model'])
    model.eval()

    for param in model.model_clip.parameters():
        param.requires_grad = False
    
    evaluate(model=model, model_clip=model_clip, data_loader=test_loader, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    print(f"程序运行时间: {elapsed_time:.6f} 秒")

