import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import random
import csv
import math
import argparse
from torchsummary import summary
import time

# Dataset class
class PCGload(Dataset):
    def __init__(self, root, resize, label, augment=True):
        super(PCGload, self).__init__()

        self.root = root
        self.resize = resize
        self.name2label = label
        self.augment = augment
        self.images, self.labels = self.load_csv('images.csv')

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('Written into CSV file:', filename)

        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        if self.augment:
            tf = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            tf = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                transforms.Resize((self.resize, self.resize)),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

# DenseNet components
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        zip_channels = self.expansion * growth_rate
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(True),
            nn.Conv2d(zip_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.cat([out, x], 1)
        return out

class SimpleDenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(SimpleDenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        out = self.features(x)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=12, compression_rate=0.5, num_classes=10, use_bottleneck=True, use_transition=True):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.compression_rate = compression_rate
        self.use_bottleneck = use_bottleneck
        self.use_transition = use_transition

        num_channels = 2 * growth_rate

        self.features = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.layer1, num_channels = self._make_dense_layer(num_channels, num_blocks[0])
        self.layer2, num_channels = self._make_dense_layer(num_channels, num_blocks[1])
        self.layer3, num_channels = self._make_dense_layer(num_channels, num_blocks[2])
        self.layer4, num_channels = self._make_dense_layer(num_channels, num_blocks[3], transition=False)
        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.AvgPool2d(4),
        )
        self.classifier = nn.Linear(num_channels, num_classes)

        self._initialize_weight()

    def _make_dense_layer(self, in_channels, nblock, transition=True):
        layers = []
        for i in range(nblock):
            if self.use_bottleneck:
                layers += [Bottleneck(in_channels, self.growth_rate)]
            else:
                layers += [SimpleDenseLayer(in_channels, self.growth_rate)]
            in_channels += self.growth_rate
        out_channels = in_channels
        if self.use_transition and transition:
            out_channels = int(math.floor(in_channels * self.compression_rate))
            layers += [Transition(in_channels, out_channels)]
        return nn.Sequential(*layers), out_channels

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# Model creation function
def create_densenet(num_blocks, growth_rate, compression_rate, num_classes, use_bottleneck, use_transition):
    return DenseNet(num_blocks, growth_rate, compression_rate, num_classes, use_bottleneck, use_transition)

# Calculate and print the model summary
def print_model_summary(model, input_size):
    summary(model, input_size)

# Training and evaluation with cost analysis
def tenepoch(model_save, train_root, test_root, augment, lr, batch_size, num_blocks, use_bottleneck, use_transition, compression_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_densenet(num_blocks, growth_rate=12, compression_rate=compression_rate, num_classes=2, 
                          use_bottleneck=use_bottleneck, use_transition=use_transition).to(device)

    # Print model summary
    print("Model Summary:")
    print_model_summary(net, (3, 32, 32))  # Assuming input size is (3, 32, 32) for CIFAR-like data

    if device == 'cuda':
        net = nn.DataParallel(net)
    torch.backends.cudnn.benchmark = True
    label = {'n': 0, 'abn': 1}
    
    train_db = PCGload(train_root, 32, label, augment=augment)
    test_db = PCGload(test_root, 32, label, augment=augment)
    trainloader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_db, batch_size=batch_size, num_workers=0)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    epoch = 50

    for e in range(epoch):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{e + 1}, {i + 1}] loss: {running_loss / 100}")
                running_loss = 0.0

        scheduler.step()

    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), model_save)

    # Evaluate the model
    correct = 0
    total = 0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    # Measure inference time
    total_time = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # Start timing
            start_time = time.time()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            end_time = time.time()

            # Accumulate total inference time
            total_time += (end_time - start_time)

            c = (predicted == labels).squeeze()
            for i in range(len(images)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print inference time and other metrics
    num_batches = len(testloader)
    avg_inference_time = total_time / num_batches
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")

    print('TP: %.2f' % class_correct[0])
    print('FN: %.2f' % (class_total[0] - class_correct[0]))
    print('TN: %.2f' % class_correct[1])
    print('FP: %.2f' % (class_total[1] - class_correct[1]))
    print('Acc: %.2f%%' % (100 * (class_correct[0] + class_correct[1]) / (class_total[0] + class_total[1])))
    print('Se: %.2f%%' % (100 * (class_correct[0]) / (class_total[0])))
    print('Sp: %.2f%%' % (100 * (class_correct[1]) / (class_total[1])))
    print('MAcc: %.2f%%' % (((100 * (class_correct[0]) / (class_total[0])) + (100 * (class_correct[1]) / (class_total[1]))) / 2))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ablation experiment with DenseNet for PCG classification.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('--model_path', type=str, default='densenet_model.pth', help='Path to save the trained model')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--no_bottleneck', action='store_true', help='Remove bottleneck layers from DenseNet')
    parser.add_argument('--no_transition', action='store_true', help='Remove transition layers from DenseNet')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[6, 12, 24, 16], help='Number of blocks in each dense layer')
    parser.add_argument('--compression_rate', type=float, default=0.5, help='Compression rate for transition layers')  # Added argument

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Run training and evaluation with the specified options
    tenepoch(
        model_save=args.model_path,
        train_root=args.train_dir,
        test_root=args.test_dir,
        augment=not args.no_augment,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        num_blocks=args.num_blocks,
        use_bottleneck=not args.no_bottleneck,
        use_transition=not args.no_transition,
        compression_rate=args.compression_rate  # Pass the new argument
    )
