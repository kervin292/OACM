import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from skimage.measure import label
from torch.utils.data import DataLoader
from utils.test_3d_patch import var_all_case_self_train
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.la import *
from networks.net_factory import net_factory
from utils.OACM_utils import context_mask, mix_loss, update_ema_variables, adjust_roi_parameters,compute_roi_parameters,apply_updated_roi_parameters


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default="../data/LA", help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='OACM', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=10000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=5, help='trained samples')
parser.add_argument('--slice_weight', type=float,  default=0.95, help='initial slice_weight')
parser.add_argument('--split', type=str,  default='train', help='datalist to use')
parser.add_argument('--slice_weight_step', type=int,  default=150, help='slice weight step')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of cut-mix
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)            #data  size
num_classes = 2
def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)


def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train1 = LAHeart1(base_dir=train_data_path,
                         split=args.split,
                         transform=transforms.Compose([
                             LARandomCrop1(patch_size, args.slice_weight),
                             LARandomRotFlip(),
                             ToTensor(),
                         ]))
    db_train2 = LAHeart2(base_dir=train_data_path,
                         split=args.split,
                         transform=transforms.Compose([
                             LARandomCrop2(patch_size, args.slice_weight),
                             LARandomRotFlip(),
                             ToTensor(),
                         ]))

    # db_train1 = KiTS1(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       KRandomCrop1(patch_size, args.slice_weight),
    #                       KRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))
    # db_train2 = KiTS2(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       KRandomCrop2(patch_size, args.slice_weight),
    #                       KRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))

    # db_train1 = LiTS1(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       LRandomCrop1(patch_size, args.slice_weight),
    #                       LRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))
    # db_train2 = LiTS2(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       LRandomCrop2(patch_size, args.slice_weight),
    #                       LRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    trainloader1 = DataLoader(db_train1, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    trainloader2 = DataLoader(db_train2, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader1) + len(trainloader2)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // (len(trainloader1) + len(trainloader2)) + 1

    last_ten_predictions = []  # Save the last ten predictions and their ROI regions

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for (_, sampled_batch1), (_, sampled_batch2) in zip(enumerate(trainloader1), enumerate(trainloader2)):
            volume_batch1, label_batch1 = sampled_batch1['image'][:args.labeled_bs], sampled_batch1['label'][:args.labeled_bs]
            volume_batch1, label_batch1 = volume_batch1.cuda(), label_batch1.cuda()
            volume_batch2, label_batch2 = sampled_batch2['image'][:args.labeled_bs], sampled_batch2['label'][:args.labeled_bs]
            volume_batch2, label_batch2 = volume_batch2.cuda(), label_batch2.cuda()
            img_a, img_b = volume_batch1[:sub_bs], volume_batch2[sub_bs:]
            lab_a, lab_b = label_batch1[:sub_bs], label_batch2[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """copy-paste"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

            # Get the last ten predictions and their ROI regions
            if len(last_ten_predictions) < 10:
                last_ten_predictions.append((outputs, img_mask))
            else:
                last_ten_predictions.pop(0)
                last_ten_predictions.append((outputs, img_mask))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()
    updated_roi_parameters = adjust_roi_parameters(last_ten_predictions)
    return last_ten_predictions  # Return the calculated ROI parameters

def self_train(args, pre_snapshot_path, self_snapshot_path, updated_roi_parameters):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()  # ema_model set

    db_train1 = LAHeart1(base_dir=train_data_path,
                         split=args.split,
                         transform=transforms.Compose([
                             LARandomCrop1(patch_size, args.slice_weight),
                             LARandomRotFlip(),
                             ToTensor(),
                         ]))
    db_train2 = LAHeart2(base_dir=train_data_path,
                         split=args.split,
                         transform=transforms.Compose([
                             LARandomCrop2(patch_size, args.slice_weight),
                             LARandomRotFlip(),
                             ToTensor(),
                         ]))

    # db_train1 = KiTS1(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       KRandomCrop1(patch_size, args.slice_weight),
    #                       KRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))
    # db_train2 = KiTS2(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       KRandomCrop2(patch_size, args.slice_weight),
    #                       KRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))

    # db_train1 = LiTS1(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       LRandomCrop1(patch_size, args.slice_weight),
    #                       LRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))
    # db_train2 = LiTS2(base_dir=train_data_path,
    #                   split=args.split,
    #                   transform=transforms.Compose([
    #                       LRandomCrop2(patch_size, args.slice_weight),
    #                       LRandomRotFlip(),
    #                       ToTensor(),
    #                   ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    trainloader1 = DataLoader(db_train1, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    trainloader2 = DataLoader(db_train2, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)

    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader1) + len(trainloader2)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // (len(trainloader1) + len(trainloader2)) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    args.mask_ratio = updated_roi_parameters['mask_ratio']  # Fixed mask_ratio using pre-training calculations

    for epoch in iterator:
        for (_, sampled_batch1), (_, sampled_batch2) in zip(enumerate(trainloader1), enumerate(trainloader2)):
            volume_batch1, label_batch1 = sampled_batch1['image'], sampled_batch1['label']
            volume_batch1, label_batch1 = volume_batch1.cuda(), label_batch1.cuda()
            volume_batch2, label_batch2 = sampled_batch2['image'], sampled_batch2['label']
            volume_batch2, label_batch2 = volume_batch2.cuda(), label_batch2.cuda()
            img_a, img_b = volume_batch1[:sub_bs], volume_batch2[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch1[:sub_bs], label_batch2[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch1[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch2[args.labeled_bs + sub_bs:]

            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
            outputs_l, _ = model(mixl_img)
            outputs_u, _ = model(mixu_img)
            loss_l = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)

            loss = loss_l + loss_u

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f' % (iter_num, loss, loss_l, loss_u))

            update_ema_variables(model, ema_model, 0.99)

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                avg_dice, avg_jc, avg_hd, avg_asd = var_all_case_self_train(model, num_classes=num_classes,
                                                                            patch_size=patch_size, stride_xy=18,
                                                                            stride_z=4)
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', avg_dice, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                writer.add_scalar('4_Var_dice/Jaccard', avg_jc, iter_num)
                writer.add_scalar('4_Var_dice/Hausdorff', avg_hd, iter_num)
                writer.add_scalar('4_Var_dice/Average_surface_distance', avg_asd, iter_num)
                model.train()

            if iter_num % 200 == 1:
                ins_width = 2
                B, C, H, W, D = outputs_l.size()
                snapshot_img = torch.zeros(size=(D, 3, 3 * H + 3 * ins_width, W + ins_width), dtype=torch.float32)

                snapshot_img[:, :, H:H + ins_width, :] = 1
                snapshot_img[:, :, 2 * H + ins_width:2 * H + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * H + 2 * ins_width:3 * H + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, W:W + ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1)  # y
                target = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_labeled' % (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1)  # y
                target = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()

if __name__ == "__main__":
    # Set snapshot paths for pre-training and self-training
    pre_snapshot_path = "./predict/OACM/LA_{}_{}_labeled/pre_train/best_model.pth".format(args.exp, args.labelnum)
    self_snapshot_path = "./predict/OACM/LA_{}_{}_labeled/self_train/best_model.pth".format(args.exp, args.labelnum)

    print("Starting OACM training.")

    # Create snapshot paths for pre-training and self-training
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')

    # Copy code files to the self-training snapshot path
    shutil.copy('train.py', self_snapshot_path)

    # Pre-training phase
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Call the pre_train function to compute last_ten_predictions
    last_ten_predictions = pre_train(args, pre_snapshot_path)

    # Call the adjust_roi_parameters function to update ROI parameters
    updated_roi_parameters = adjust_roi_parameters(last_ten_predictions)

    # Pass the updated ROI parameters to the apply_updated_roi_parameters function
    args = apply_updated_roi_parameters(updated_roi_parameters, args)

    # Self-training phase
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Call the self_train function for self-training
    self_train(args, pre_snapshot_path, self_snapshot_path, updated_roi_parameters)


