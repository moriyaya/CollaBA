import argparse
import torch
import cv2
import glob
import os
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.mirnetv2_arch import MIRNetv2
from basicsr.archs.mirnetv2atten_arch import MIRNetv2atten
from basicsr.archs.mirnetv2ueatten_arch import MIRNetv2ueatten
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from ipdb import set_trace


def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/data2/zyyue/dataset/RELLISUR-Dataset/Test_crop/LLLR/', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealMIRUEATTENSRGAN_x2plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus'
              'RealESRGANv2-anime-xsx2 | RealESRGANv2-animevideo-xsx2-nousm | RealESRGANv2-animevideo-xsx2'
              'RealESRGANv2-anime-xsx4 | RealESRGANv2-animevideo-xsx4-nousm | RealESRGANv2-animevideo-xsx4'))
    parser.add_argument('-o', '--output', type=str, default='results_RealMIRUEATTENSRGAN_x2_wgn_20000_woautoloss', help='Output folder')
    parser.add_argument('-oa', '--output_atten', type=str, default='results_attenmap_RealMIRUEATTENSRGAN_x2_wgn_20000_woautoloss', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealMIRSRGAN_x2plus']:  # x2 MIRNetv2 model with 6 blocks
        model = MIRNetv2(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=2)
        netscale = 2
    elif args.model_name in ['RealMIRSRGAN_x4plus']:  # x4 MIRNetv2 model with 6 blocks
        model = MIRNetv2(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=4)
        netscale = 4
    elif args.model_name in ['RealMIRATTENSRGAN_x2plus']:  # x2 MIRNetv2atten model
        model = MIRNetv2atten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=2)
        netscale = 2
    elif args.model_name in ['RealMIRATTENSRGAN_x4plus']:  # x4 MIRNetv2atten model
        model = MIRNetv2atten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=4)
        netscale = 4
    elif args.model_name in ['RealMIRUEATTENSRGAN_x2plus']:  # x2 MIRNetv2ueatten model
        model = MIRNetv2ueatten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3,width=2, scale=2)
        netscale = 2
    elif args.model_name in ['RealMIRUEATTENSRGAN_x4plus']:  # x4 MIRNetv2ueatten model
        model = MIRNetv2ueatten(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3,width=2, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in [
            'RealESRGANv2-anime-xsx2', 'RealESRGANv2-animevideo-xsx2-nousm', 'RealESRGANv2-animevideo-xsx2'
    ]:  # x2 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    elif args.model_name in [
            'RealESRGANv2-anime-xsx4', 'RealESRGANv2-animevideo-xsx4-nousm', 'RealESRGANv2-animevideo-xsx4'
    ]:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    #model_path = os.path.join('/data2/zyyue/Real-ESRGAN-master/experiments/train_RealMIRATTENSRGANx4plus_our_100k_lr2e-4_gn_new/models/', args.model_name + '_5000.pth')
    # model_path = os.path.join(
    #     '/data2/zyyue/Real-ESRGAN-master/experiments/train_RealMIRATTENSRGANx2plus_our_100k_lr2e-4_gn/models/',
    #     args.model_name + '_65000.pth')
    #model_path= "/data2/zyyue/Real-ESRGAN-master/experiments/train_RealMIRUEATTENSRGANx2plus_our_100k_lr2e-4_wgn_glossauto_add_ue_tv/models/net_g_20000.pth"
    model_path = "/data2/zyyue/Real-ESRGAN-master/experiments/train_RealMIRUEATTENSRGANx4plus_our_100k_lr2e-4_wgn_glossauto_add_ue_tv_/models/net_g_5000.pth"

# model_path= "/data2/zyyue/Real-ESRGAN-master/experiments/train_RealMIRUEATTENSRGANx2plus_our_100k_lr1e-4_25w35w45w_wgn_glossauto_add_ue_tv_plot2/models/net_g_25000.pth"
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half,
        use_grayatten=False,
        use_ue_atten=True)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    time_sum = 0

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                # torch.cuda.synchronize()
                # time_start = time.time()
                output, _, time_sum_tmp, atten_map = upsampler.enhance(img, outscale=args.outscale)
                # torch.cuda.synchronize()
                # time_end = time.time()
                # time_sum_tmp = time_end - time_start
                time_sum += time_sum_tmp
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            save_path = os.path.join(args.output, f'{imgname}{args.suffix}.{extension}')
            attenmap_save_path = os.path.join(args.output_atten, f'{imgname}{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)
            cv2.imwrite(attenmap_save_path, output)

    print(time_sum/len(paths))


if __name__ == '__main__':
    main()
