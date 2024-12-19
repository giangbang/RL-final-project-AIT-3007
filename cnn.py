import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 5, 3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 5, 3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
    def get_output_dim(self, size):
        # Tính kích thước output sau khi qua CNN
        # Conv2d(kernel=3): h,w -> h-2,w-2
        # MaxPool2d(2): h,w -> h//2,w//2
        h, w = size
        h = (h - 2) - 2  # Sau 2 lớp Conv2d
        w = (w - 2) - 2
        h = h // 2  # Sau MaxPool2d
        w = w // 2
        return h * w * 5  # 5 là số channels output

    def forward(self, x):
        # x shape: [..., H, W, 5]
        orig_shape = x.shape[:-3]
        h, w = x.shape[-3:-1]
        
        x = x.view(-1, h, w, 5)
        x = x.permute(0, 3, 1, 2)  # [..., 5, H, W]
        x = self.cnn(x)
        x = x.reshape(*orig_shape, -1)  # Flatten CNN output
        return x