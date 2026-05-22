import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import sys

from models import *

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Prediction')
    parser.add_argument('--image', default='', type=str, help='path to input image')
    parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth', type=str, help='path to checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==> Using device: {device}')
    
    # 1. 모델 빌드
    print('==> Building model (ResNet-50)..')
    net = ResNet50()
    
    # 2. 체크포인트 로드
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint file not found at '{checkpoint_path}'")
        print("Please train the model first or specify a valid checkpoint path using --checkpoint.")
        print("Example: python main.py")
        sys.exit(1)
        
    print(f'==> Loading checkpoint from {checkpoint_path}..')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # state_dict 키 처리 (DataParallel을 사용하여 저장되었을 경우 'module.' 접두사 제거)
    state_dict = checkpoint['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict)
    net = net.to(device)
    net.eval()
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
               
    # 3. 입력 이미지 전처리 및 추론
    if args.image == '':
        print("No image file specified via --image. Attempting to load a sample from CIFAR-10 test set..")
        try:
            import torchvision
            # CIFAR-10 테스트셋 로드 (학습시 이미 다운로드 받았을 것으로 가정, 없을 시 다운로드)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True)
            
            # 첫 번째 샘플 가져오기 (0번째)
            sample_img, sample_label = testset[0]
            print(f"Successfully loaded a sample from CIFAR-10 test set (Ground Truth: {classes[sample_label]})")
            
            # 임시 파일로 저장하여 사용자가 확인할 수 있도록 함
            sample_img_path = './demo_sample.png'
            sample_img.save(sample_img_path)
            print(f"Saved the sample image to '{sample_img_path}' for reference.")
            
            # CIFAR-10용 Normalize 변환 적용 (32x32 크기이므로 resize 불필요)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            inputs = transform_test(sample_img).unsqueeze(0).to(device)
            target_name = f"CIFAR-10 Test Sample 0 (Ground Truth: {classes[sample_label]})"
            
        except Exception as e:
            print(f"Failed to load CIFAR-10 test sample: {e}")
            print("Falling back to random dummy input (3, 32, 32).")
            inputs = torch.randn(1, 3, 32, 32).to(device)
            target_name = "Random Dummy Tensor"
    else:
        if not os.path.exists(args.image):
            print(f"Error: image file not found at '{args.image}'")
            sys.exit(1)
            
        # 사용자 이미지 입력 시 32x32로 크기 조정 및 정규화
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        try:
            image = Image.open(args.image).convert('RGB')
        except Exception as e:
            print(f"Error opening image '{args.image}': {e}")
            sys.exit(1)
            
        inputs = transform_test(image).unsqueeze(0).to(device)
        target_name = f"User Image: {args.image}"
        
    # 4. 추론 수행
    with torch.no_grad():
        outputs = net(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
        
    print(f"\n[Inference Result for {target_name}]")
    print(f"Predicted class: {classes[predicted.item()]} (Confidence: {confidence.item()*100:.2f}%)")
    print("\nAll class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  {classes[i]:<10}: {prob.item()*100:.2f}%")

if __name__ == '__main__':
    main()
