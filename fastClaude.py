import os
import torch
import pandas as pd
import ast
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torchvision
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

class CocoSubsetDataset(Dataset):
    def __init__(self, csv_path, images_path, transforms=None):
        self.transforms = transforms
        self.images_path = images_path
        self.df = pd.read_csv(csv_path)
        self.df['bbox'] = self.df['bbox'].apply(ast.literal_eval)
        
        self.category_to_id = {
            'person': 1,
            'cat': 2,
            'dog': 3
        }
        
    def __getitem__(self, idx):
        img_annots = self.df[self.df['image_id'] == self.df['image_id'].unique()[idx]]
        img_path = os.path.join(self.images_path, img_annots['image'].iloc[0])
        img = read_image(img_path)
        
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        boxes = []
        labels = []
        
        for _, row in img_annots.iterrows():
            bbox = row['bbox']
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])
            labels.append(self.category_to_id[row['label']])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        img = tv_tensors.Image(img)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.df['image_id'].unique())

class ObjectDetectionTrainer:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.model = self._get_model()
        self.model.to(device)
        
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.1
        )
        
        # Fixed: Initialize metrics as lists
        self.metrics = {
            'losses': [],
            'maps': [],
            'class_aps': []  # Will store dictionaries of per-epoch APs
        }
        
        self.class_names = {1: 'person', 2: 'cat', 3: 'dog'}
    
    def _get_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model
    
    def train_epoch(self, data_loader):
        self.model.train()
        epoch_loss = 0
        
        with tqdm(data_loader, desc="Training") as pbar:
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                
                epoch_loss += losses.item()
                pbar.set_postfix({'loss': losses.item()})
        
        return epoch_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        self.model.eval()
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Evaluating"):
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                
                for output, target in zip(outputs, targets):
                    predictions.append({
                        'boxes': output['boxes'].cpu(),
                        'scores': output['scores'].cpu(),
                        'labels': output['labels'].cpu()
                    })
                    ground_truths.append({
                        'boxes': target['boxes'],
                        'labels': target['labels']
                    })
        
        return self._calculate_metrics(predictions, ground_truths)
    
    def _calculate_metrics(self, predictions, ground_truths):
        class_scores = {i: {'scores': [], 'labels': []} for i in range(1, self.num_classes)}
        
        for pred, gt in zip(predictions, ground_truths):
            for class_id in range(1, self.num_classes):
                # Get predictions for this class
                class_mask_pred = pred['labels'] == class_id
                class_mask_gt = gt['labels'] == class_id
                
                # Add positive examples
                for _ in range(class_mask_gt.sum()):
                    class_scores[class_id]['labels'].append(1)
                    if class_mask_pred.sum() > 0:
                        class_scores[class_id]['scores'].append(
                            pred['scores'][class_mask_pred].max().item()
                        )
                    else:
                        class_scores[class_id]['scores'].append(0.0)
                
                # Add negative examples
                other_scores = pred['scores'][pred['labels'] != class_id]
                for score in other_scores:
                    class_scores[class_id]['labels'].append(0)
                    class_scores[class_id]['scores'].append(score.item())
        
        # Calculate AP for each class
        aps = {}
        for class_id in range(1, self.num_classes):
            if len(class_scores[class_id]['scores']) > 0:
                ap = average_precision_score(
                    class_scores[class_id]['labels'],
                    class_scores[class_id]['scores']
                )
                aps[class_id] = ap
        
        return {
            'class_aps': aps,
            'map': np.mean(list(aps.values()))
        }
    
    def plot_metrics(self, save_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epochs = range(1, len(self.metrics['losses']) + 1)
        
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Loss Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.metrics['losses'], 'r-', label='Training Loss')
        ax1.set_title('Model Loss over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # 2. mAP Plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.metrics['maps'], 'b-', label='mAP')
        ax2.set_title('Mean Average Precision over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('mAP')
        ax2.grid(True)
        ax2.legend()
        
        # 3. Per-class AP Plot
        ax3 = fig.add_subplot(gs[1, :])
        for class_id in range(1, self.num_classes):
            class_aps = [epoch_aps.get(class_id, 0) for epoch_aps in self.metrics['class_aps']]
            ax3.plot(epochs, class_aps, label=f'{self.class_names[class_id]} AP')
        
        ax3.set_title('Per-class Average Precision over Epochs')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Average Precision')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f'metrics_{timestamp}.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    
    def save_model(self, save_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f'model_{timestamp}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)
        return path
    
    def train(self, train_loader, val_loader, num_epochs, save_dir):
        best_map = 0.0
        best_model_path = None
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            epoch_loss = self.train_epoch(train_loader)
            self.metrics['losses'].append(epoch_loss)
            
            # Evaluate
            eval_metrics = self.evaluate(val_loader)
            self.metrics['maps'].append(eval_metrics['map'])
            self.metrics['class_aps'].append(eval_metrics['class_aps'])
            
            print(f"Loss: {epoch_loss:.4f}, mAP: {eval_metrics['map']:.4f}")
            for class_id, ap in eval_metrics['class_aps'].items():
                print(f"{self.class_names[class_id]} AP: {ap:.4f}")
            
            # Save best model
            if eval_metrics['map'] > best_map:
                best_map = eval_metrics['map']
                best_model_path = self.save_model(save_dir)
                print(f"New best model saved with mAP: {best_map:.4f}")
            
            self.lr_scheduler.step()
            self.plot_metrics(save_dir)
        
        print(f"\nTraining completed!")
        print(f"Best mAP: {best_map:.4f}")
        print(f"Best model path: {best_model_path}")
        
        # Save final model
        final_path = self.save_model(save_dir)
        return final_path, best_model_path

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def visualize_predictions(model, image_path, device, transform, confidence_threshold=0.5):
    image = read_image(image_path)
    image_transformed = transform(image)
    image_transformed = image_transformed[:3, ...].to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model([image_transformed])[0]
    
    image = image[:3, ...]
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    
    pred_scores = predictions["scores"] > confidence_threshold
    pred_boxes = predictions["boxes"][pred_scores].long()
    pred_labels = predictions["labels"][pred_scores]
    
    id_to_label = {1: 'person', 2: 'cat', 3: 'dog'}
    pred_label_texts = [
        f"{id_to_label[label.item()]}: {score:.2f}" 
        for label, score in zip(pred_labels, predictions["scores"][pred_scores])
    ]
    
    output_image = draw_bounding_boxes(
        image, 
        pred_boxes,
        pred_label_texts,
        colors="red"
    )
    
    return output_image

def main():
    # Configuration
    # TRAIN_IMAGES_PATH = 'D:/Download/JDownloader/MSCOCO/images/train2017'
    # FILTERED_PATH = 'D:/Projetos/pythonlib/filtered-coco-dataset'
    # WORKING_DIR = 'D:/Projetos/pythonlib/working'
    # CSV_PATH = os.path.join(FILTERED_PATH, 'filtered_coco.csv')  # Fixed: Use CSV path instead of directory
    TRAIN_IMAGES_PATH = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
    WORKING_DIR = '/kaggle/working'  
    FILTERED_PATH = '/kaggle/input/filtered-coco-dataset'
    CSV_PATH = os.path.join(FILTERED_PATH, 'filtered_coco.csv')
    NUM_CLASSES = 4  # background + person + cat + dog
    NUM_EPOCHS = 3
    BATCH_SIZE = 2
    
    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # Verify paths exist
    if not os.path.exists(TRAIN_IMAGES_PATH):
        raise FileNotFoundError(f"Training images directory not found: {TRAIN_IMAGES_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    try:
        dataset = CocoSubsetDataset(
            CSV_PATH,  # Fixed: Use CSV path
            TRAIN_IMAGES_PATH,
            get_transform(train=True)
        )
        dataset_test = CocoSubsetDataset(
            CSV_PATH,  # Fixed: Use CSV path
            TRAIN_IMAGES_PATH,
            get_transform(train=False)
        )
    except Exception as e:
        print(f"Error creating datasets: {str(e)}")
        raise
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Split dataset
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset_test, indices[train_size:])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize trainer
    trainer = ObjectDetectionTrainer(NUM_CLASSES, device)
    
    # Train model
    try:
        final_model_path = trainer.train(
            train_loader,
            val_loader,
            NUM_EPOCHS,
            WORKING_DIR
        )
        print(f"Training completed! Final model saved to: {final_model_path}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program failed with error: {str(e)}")
        import traceback
        traceback.print_exc()