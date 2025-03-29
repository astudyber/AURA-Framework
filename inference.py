import os
import time
import argparse
from utils import *
from model import *
from tqdm import tqdm
from pathlib import Path


def get_opts(known=False):
    # 1. init opts
    opts = argparse.ArgumentParser()
    opts.add_argument('--folder', default='test/', help='input folder')
    opts.add_argument('--output', default='results/', help='output folder')

    # 2. return opts
    return opts.parse_known_args()[0] if known else opts.parse_args()


if __name__ == '__main__':
    opts = get_opts()
    # opts.--folder
    test = get_filenames(opts.folder)
    BATCH_TEST = 1
    test_dataset = LoadData(root=opts, rgb_files=test, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_TEST, shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
    model = ResCSPNet(enc_chs=Configs.encoder, dec_chs=Configs.decoder, out_ch=Configs.out_ch, out_sz=Configs.out_sz)
    # 加载权重文件
    device = torch.device("cuda")
    pretrained_weights_path = r".\model.pt"
    model = torch.load(pretrained_weights_path, map_location=device, weights_only=False)
    model = model.to(device)

    SUBMISSION_PATH = opts.output
    runtime = []

    see_map1 = {}
    see_map2 = {}

    cnt = 0
    model.eval()
    with torch.no_grad():
        for (rgb_batch, rgb_name) in tqdm(test_loader):
            rgb_batch = rgb_batch.to(device)
            real_name = Path(rgb_name[0]).name
            real_name = real_name.split('/')[-1].replace('.png', '')
            rgb_name = rgb_name[0].split('/')[-1].replace('.png', '')

            st = time.time()
            recon_raw = model(rgb_batch)
            tt = time.time() - st
            runtime.append(tt)

            recon_raw = recon_raw[0].detach().cpu().permute(1, 2, 0).numpy()
            rgb_batch = rgb_batch[0].detach().cpu().permute(1, 2, 0).numpy()

            see_map1[rgb_name] = (rgb_batch)
            see_map2[rgb_name] = postprocess_raw(demosaic(recon_raw))

            ## save as np.uint16
            assert recon_raw.shape[-1] == 4
            recon_raw = (recon_raw * 1024).astype(np.uint16)
            np.save(SUBMISSION_PATH + real_name + '.npy', recon_raw)
            cnt += 1

    print(np.mean(runtime))