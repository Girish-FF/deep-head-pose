import sys, os, argparse
from time import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import pandas as pd
import datasets, hopenet, utils
import hopenetlite_v2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    prog_start = time()
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    # ResNet50 structure
    # model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading snapshot.')
    # Load snapshot
    # saved_state_dict = torch.load(snapshot_path)
    # model.load_state_dict(saved_state_dict)
    model = hopenetlite_v2.HopeNetLite()
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict, strict=False)
    # net.eval()

    print('Loading data.')

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000_ds':
        pose_dataset = datasets.AFLW2000_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'custom':
        pose_dataset = datasets.custom(args.data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2)

    # model.cuda(gpu)
    model.to("cpu")

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to("cpu")

    # yaw_error = .0
    # pitch_error = .0
    # roll_error = .0

    l1loss = torch.nn.L1Loss(size_average=False)
    details = {"Name":[], "Infer_Time":[], "Yaw":[], "Pitch":[], "Roll":[]}
    for i, (images, name) in enumerate(test_loader):
        # images = Variable(images).cuda(gpu)
        images = Variable(images).to("cpu")
        # total += cont_labels.size(0)

        # label_yaw = cont_labels[:,0].float()
        # label_pitch = cont_labels[:,1].float()
        # label_roll = cont_labels[:,2].float()
        start = time()
        yaw, pitch, roll = model(images)
        end = time()

        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        # Mean absolute error
        # yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
        # pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
        # roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

        infer_time = (end-start)*1000

        details['Name'].append(name[0])
        details['Infer_Time'].append(infer_time)
        details['Yaw'].append(yaw_predicted[0].numpy())
        details['Pitch'].append(pitch_predicted[0].numpy())
        details['Roll'].append(roll_predicted[0].numpy())

        # Save first image in batch with pose cube or axis.
        if args.save_viz:
            name = name[0]
            if args.dataset == 'BIWI':
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + '_rgb.png'))
            else:
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg') )
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB) 
                cv2_img = cv2.resize(cv2_img, (224,224))
            if args.batch_size == 1:
                # true_val = 'y %.2f, p %.2f, r %.2f' % (label_yaw, label_pitch, label_roll)
                pred_val = 'y %.2f, p %.2f, r %.2f' % (yaw_predicted, pitch_predicted, roll_predicted)
                # error_string = 'y %.2f, p %.2f, r %.2f' % (torch.sum(torch.abs(yaw_predicted - label_yaw)), torch.sum(torch.abs(pitch_predicted - label_pitch)), torch.sum(torch.abs(roll_predicted - label_roll)))
                # cv2.putText(cv2_img, true_val, (30, cv2_img.shape[0]- 45), fontFace=1, fontScale=1, color=(0,255,0), thickness=2)
                cv2.putText(cv2_img, pred_val, (10, cv2_img.shape[0]- 30), fontFace=1, fontScale=0.8, color=(0,0,255), thickness=1)
                # cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0]- 15), fontFace=1, fontScale=1, color=(255,0,0), thickness=2)
            # utils.plot_pose_cube(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
            # cv2_img2 = cv2_img.copy()
            utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx = 100, tdy= 100, size=50)
            # utils.draw_axis(cv2_img2, label_yaw[0], label_pitch[0], label_roll[0], tdx = 200, tdy= 200, size=100)
            # img = Image.fromarray(cv2_img)
            # img.show("img")
            # cv2.imwrite(os.path.join('C:\\Users\\Girish\\Desktop\\FF-Projects\\deep-head-pose\\output\\images', name + '.jpg'), cv2_img)
            os.makedirs('C:\\Users\\Girish\\Desktop\\FF-Projects\\deep-head-pose\\output\\custom_images_hopenetlite', exist_ok=True)
            # img.save(os.path.join('C:\\Users\\Girish\\Desktop\\FF-Projects\\deep-head-pose\\output\\images', name + '.jpg'))
            # print("Visualization Saved")
            # cv2.waitKey(0)
            # fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # ax[0].imshow(cv2_img2)
            # ax[0].axis('off')
            # ax[0].set_title(f'True')

            plt.imshow(cv2_img)
            plt.axis('off')
            plt.title(f'Prediction')
            # plt.imshow(cam_image)
            plt.tight_layout()
            plt.axis('off')
            # plt.title(f'{classes[cam_label]}-{cam_prob.values.item():.4f}\n{classes[pred_lbl]}-{probability:.4f}')
            # plt.suptitle(f"Original Class: {folders[true_lbl]}")
            plt.savefig(os.path.join('C:\\Users\\Girish\\Desktop\\FF-Projects\\deep-head-pose\\output\\custom_images_hopenetlite', name + '.jpg'), bbox_inches='tight')
            plt.close()

    # print('Test error in degrees of the model on the ' + str(total) +
    # ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw_error / total,
    # pitch_error / total, roll_error / total))

    print(f"Average inference time by the model is {np.mean(details['Infer_Time'])} miliseconds")        
    # # Create a DataFrame with the common values
    df_common_paths = pd.DataFrame(details)

    # Save the DataFrame to a CSV file
    csv_file_path = 'output//hopenetlite_predictions_cpu.csv'
    df_common_paths.to_csv(csv_file_path, index=False)
    print(f"Total time taken to run the script: {time()-prog_start} seconds.")
