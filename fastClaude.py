import os
import torch
import pandas as pd
import ast
import torchvision
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
import utils
from engine import train_one_epoch, evaluate

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class CocoSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_path, transforms=None):
        self.transforms = transforms
        self.images_path = images_path
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        # Convert string representation of bbox to list
        self.df['bbox'] = self.df['bbox'].apply(ast.literal_eval)
        
        # Create category to id mapping
        self.category_to_id = {
            'person': 1,
            'cat': 2,
            'dog': 3
        }
        
    def __getitem__(self, idx):
        # Get all annotations for this image
        img_annots = self.df[self.df['image_id'] == self.df['image_id'].unique()[idx]]
        
        # Load image
        img_path = os.path.join(self.images_path, img_annots['image'].iloc[0])
        img = read_image(img_path)
        
        # If image is grayscale, convert to RGB
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        # Get boxes and labels
        boxes = []
        labels = []
        
        for _, row in img_annots.iterrows():
            # Convert COCO bbox [x,y,width,height] to [x1,y1,x2,y2]
            bbox = row['bbox']
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])
            labels.append(self.category_to_id[row['label']])
        
        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Prepare target dict
        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.df['image_id'].unique())

def get_model_detection(num_classes):
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Get number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Training setup
TRAIN_IMAGES_PATH = 'D:/Download/JDownloader/MSCOCO/images/train2017'
FILTERED_PATH = 'D:/Projetos/pythonlib/filtered-coco-dataset'
WORKING_DIR = 'D:/Projetos/pythonlib/working'
CSV_PATH = os.path.join(FILTERED_PATH, 'filtered_coco.csv')

# Number of classes (background + person + cat + dog)
num_classes = 4

# Create train and test datasets
dataset = CocoSubsetDataset(CSV_PATH, TRAIN_IMAGES_PATH, get_transform(train=True))
dataset_test = CocoSubsetDataset(CSV_PATH, TRAIN_IMAGES_PATH, get_transform(train=False))

# Split dataset
indices = torch.randperm(len(dataset)).tolist()
train_size = int(0.8 * len(dataset))  # 80% for training
dataset = torch.utils.data.Subset(dataset, indices[:train_size])
dataset_test = torch.utils.data.Subset(dataset_test, indices[train_size:])

# Create data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# Initialize model, optimizer and scheduler
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_detection(num_classes)
model.to(device)

# Optimize all parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 2

for epoch in range(num_epochs):
    # Train for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    
    # Update learning rate
    lr_scheduler.step()
    
    # Evaluate on test dataset
    #evaluate(model, data_loader_test, device=device)

print("Training completed!")

# Visualization function
def visualize_predictions(model, image_path, device, transform):
    image = read_image(image_path)
    
    # Transform image
    image_transformed = transform(image)
    image_transformed = image_transformed[:3, ...].to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model([image_transformed])
        pred = predictions[0]
    
    # Convert image for visualization
    image = image[:3, ...]  # Remove alpha channel if present
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    
    # Draw bounding boxes
    pred_scores = pred["scores"] > 0.5
    pred_boxes = pred["boxes"][pred_scores].long()
    pred_labels = pred["labels"][pred_scores]
    
    # Convert numeric labels to text
    id_to_label = {1: 'person', 2: 'cat', 3: 'dog'}
    pred_label_texts = [f"{id_to_label[label.item()]}: {score:.2f}" 
                       for label, score in zip(pred_labels, pred["scores"][pred_scores])]
    
    # Draw boxes
    output_image = draw_bounding_boxes(
        image, 
        pred_boxes,
        pred_label_texts,
        colors="red"
    )
    
    return output_image