import cv2
import numpy as np
import argparse
from PIL import Image
import torchvision
import torch.nn.parallel
import torch.optim
from ops.models import TSN
from ops.transforms import * 
from torch.nn import functional as F
import os


if torch.cuda.is_available():
    print("\nYou have an \"Nvidia ",torch.cuda.get_device_name(0),"\" (Cuda enabled GPU)")
    GPU_FLAG = input("USE GPU (y/n) : ")
else:
    print("NO GPU found, Running on CPU!")
    GPU_FLAG = 'n'

parser = argparse.ArgumentParser(description="TSM Testing on real time!!")
parser.add_argument('-f',type=str,help='Provide a video!!')
parser.add_argument('--arch',type=str,help='provide architecture [mobilenetv2,resnet50]')

print()
print('======>>>>> Loading model ... Please wait ...')
#just adding some comments to check git 
def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

args = parser.parse_args()

this_weights='checkpoint/TSM_ucfcrime_RGB_resnet50_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar'

is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
modality = 'RGB'
if 'RGB' in this_weights:
	modality = 'RGB'

# Get dataset categories.
categories = ['Normal Activity','Abnormal Activity']
num_class = len(categories)
this_arch = args.arch

net = TSN(num_class, 1, modality,
              base_model=this_arch,
              consensus_type='avg',
              img_feature_dim='225',
              #pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              )
if GPU_FLAG is 'y':
    checkpoint = torch.load(this_weights)
else:
    checkpoint = torch.load(this_weights,map_location=torch.device('cpu'))


checkpoint = checkpoint['state_dict']

# base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
               }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)
net.load_state_dict(base_dict)

if GPU_FLAG is 'y':
    net.cuda().eval()
else:
    net.eval()


#net.eval()
transform=torchvision.transforms.Compose([
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ])



WINDOW_NAME = 'Video Action Recognition'

def doInferecing(cap):
# set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    i_frame = -1
    count = 0
    print("Ready!")
    
    while cap.isOpened():
        count+=1
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
       
            
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            if GPU_FLAG is 'y':
                input1 = img_tran.view(-1, 3, img_tran.size(1),
                img_tran.size(2)).unsqueeze(0).cuda()
            else:
                input1 = img_tran.view(-1, 3, img_tran.size(1),
                img_tran.size(2)).unsqueeze(0)
            
            
            
            input = input1
            #input = img_tran.view(-1, 3, img_tran.size(1),
            #img_tran.size(2)).unsqueeze(0)
            with torch.no_grad():
               logits = net(input)
               h_x = torch.mean(F.softmax(logits, 1), dim=0).data
               print(h_x)
               pr, li = h_x.sort(0, True)
               probs = pr.tolist()
               idx = li.tolist()
               #print(probs)
               print(count,'-',categories[idx[0]],'Prob: ',probs[0])

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255
        
        if categories[idx[0]] == 'Abnormal Activity':
            R = 255
            G = 0  
            #print('\007')  
        else:
            R = 0
            G = 255
            
        cv2.putText(label, 'EVENT: ' + categories[idx[0]],
                   (10, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, int(G), int(R)), 2)
        
        cv2.putText(label, 'Confidence: {:.2f}%'.format(probs[0]*100,'%'),
                    (width - 250 , int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, int(G), int(R)), 2)
                  
        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)
 
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
def main():
    args = vars(parser.parse_args())
    
    if not args.get('f', False):
        print("Openinig camera...")
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture('http://192.168.43.1:8080/video')
        
    else:
        print("loading Video...")
        cap = cv2.VideoCapture(args['f'])
    doInferecing(cap)


        


main()
