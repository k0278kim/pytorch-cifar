import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import sys
import time

from models import *

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Prediction')
    parser.add_argument('--image', default='', type=str, help='path to input image')
    parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth', type=str, help='path to checkpoint')
    parser.add_argument('--batch-size', default=100, type=int, help='batch size for evaluation')
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
               
    # 3. 단일 이미지 추론 모드
    if args.image != '':
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
        
        # 추론 및 시간 측정 수행
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = net(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000 # ms
        
        print(f"\n[Inference Result for User Image: '{args.image}']")
        print(f"Predicted class: {classes[predicted.item()]} (Confidence: {confidence.item()*100:.2f}%)")
        print(f"Inference Time : {inference_time:.2f} ms")
        print("\nAll class probabilities:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  {classes[i]:<10}: {prob.item()*100:.2f}%")
            
    # 4. 전체 CIFAR-10 테스트 데이터셋 일괄 추론 및 평가 모드
    else:
        print("No image file specified. Evaluating the model on the ENTIRE CIFAR-10 test dataset..")
        
        # CIFAR-10 테스트 데이터셋 정규화 트랜스폼
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        try:
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        except Exception as e:
            print(f"Error loading CIFAR-10 dataset: {e}")
            sys.exit(1)
            
        correct = 0
        total = 0
        
        # 클래스별 통계 데이터 구조 초기화
        class_correct = [0] * 10
        class_total = [0] * 10
        
        print(f"Starting inference on {len(testset)} images (Batch Size: {args.batch_size})..")
        
        # 전체 dataset 평가 시간 측정 시작
        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 클래스별 개별 정확도 기록을 위한 안전한 매칭 카운팅
                c = (predicted == targets)
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_correct[label] += int(c[i].item())
                    class_total[label] += 1
                
                # 5개 배치 단위 또는 마지막 배치 도달 시 가벼운 진행 상황 로그 출력
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(testloader):
                    progress = (batch_idx + 1) / len(testloader) * 100
                    print(f"Progress: {progress:>6.2f}% | Batch [{batch_idx+1}/{len(testloader)}] | Current Accum. Acc: {100.*correct/total:.2f}%")
        
        # 전체 dataset 평가 시간 측정 종료
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = (total_time / total) * 1000 if total > 0 else 0 # ms per image
        
        # 5. 최종 결과 리포트 출력
        total_acc = 100. * correct / total
        print("\n" + "="*55)
        print(f"🏆 [Evaluation Result] Total Test Accuracy: {total_acc:.2f}% ({correct}/{total})")
        print(f"⏱️  [Runtime Statistics]")
        print(f"  - Total Elapsed Time : {total_time:.2f} seconds")
        print(f"  - Avg Time per Image : {avg_time:.2f} ms")
        print("="*55)
        print(f" {'Class':<12} | {'Accuracy':<10} | {'Correct/Total':<15}")
        print("-"*55)
        for i in range(10):
            acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            correct_str = f"{class_correct[i]}/{class_total[i]}"
            print(f"  {classes[i]:<10} | {acc:>8.2f}% | {correct_str:<15}")
        print("="*55)

if __name__ == '__main__':
    main()
