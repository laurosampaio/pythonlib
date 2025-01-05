import os
import torch
import pandas as pd
import numpy as np
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset, DataLoader

############################
# 1) Define the Dataset
############################
class FilteredCocoDataset(Dataset):
    def __init__(self, csv_path, images_dir, transforms=None):
        """
        Args:
            csv_path (str): Path to 'filtered_coco.csv'.
            images_dir (str): Directory containing the images.
            transforms (callable, optional): Optional transform to be applied on an image sample.
        """
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Map "category_id" from {1,17,18} to contiguous [1,2,3] if needed
        # Or keep {1,17,18} directly, as long as 'num_classes' accounts for it.
        # Below is an example of a simple map to [1..3].
        # If you prefer to directly use 1,17,18, then adjust num_classes accordingly.
        self.id_map = {1: 1, 17: 2, 18: 3}  # background will be 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        
        # Retrieve image
        img_path = os.path.join(self.images_dir, record["image"])
        image = Image.open(img_path).convert("RGB")
        
        # Convert string bbox [x, y, w, h] into a Python list
        # record["bbox"] is something like "[344.18, 93.86, 48.47, 100.01]"
        # Evaluate or parse it safely:
        bbox = eval(record["bbox"])  # or use json.loads if you prefer

        # Convert [x, y, w, h] → [xmin, ymin, xmax, ymax]
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        
        # Map category_id into contiguous label if needed
        cat_id = record["category_id"]
        label = self.id_map.get(cat_id, 0)  # default 0 for background
        labels = torch.tensor([label], dtype=torch.int64)

        # Prepare target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        # Convert to torchvision Image (if using transforms v2 functional)
        # or you can still use standard transforms.ToTensor().
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

############################
# 2) Define the Transforms
############################
# (Same style as get_transform from the attached file [1], but no mask transforms needed.)
from torchvision.transforms import v2 as T

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

############################
# 3) Define Model
############################
# We only do object detection, no instance segmentation
def get_detection_model(num_classes):
    """
    Returns a Faster R-CNN model with the specified number of classes.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

############################
# 4) Create DataLoaders
############################
TRAIN_IMAGES_PATH = 'D:/Download/JDownloader/MSCOCO/images/train2017'
FILTERED_PATH = 'D:/Projetos/pythonlib/working'
CSV_PATH = os.path.join(FILTERED_PATH, 'filtered_coco.csv')

# Our classes: background=0, person=1, cat=2, dog=3  → num_classes=4
num_classes = 4

dataset = FilteredCocoDataset(
    csv_path=CSV_PATH,
    images_dir=TRAIN_IMAGES_PATH,
    transforms=get_transform(train=True)
)
dataset_test = FilteredCocoDataset(
    csv_path=CSV_PATH,
    images_dir=TRAIN_IMAGES_PATH,
    transforms=get_transform(train=False)
)

# For simplicity, we might split dataset randomly
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
split_idx = int(0.8 * len(indices))  # 80/20 split
dataset_train = torch.utils.data.Subset(dataset, indices[:split_idx])
dataset_val   = torch.utils.data.Subset(dataset_test, indices[split_idx:])

data_loader = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda batch: tuple(zip(*batch))
)
data_loader_test = DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch))
)

############################
# 5) Train/Evaluate Model
############################
import utils
from engine import train_one_epoch, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


model = get_detection_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 2
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)

print("Training complete!")
