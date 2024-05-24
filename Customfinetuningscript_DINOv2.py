import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import evaluate
from torchvision.transforms.functional import to_tensor
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec

def save_checkpoint(model, optimizer, scheduler, epoch, file_path="C:\\Users\\Stell\\Desktop\\DVA 309\\checkpointDINO3.pth"):
    """Save model and optimizer states to a file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, file_path)
    print(f"Checkpoint saved at epoch {epoch}.")

def load_checkpoint(file_path, model, optimizer, scheduler):
    """Load model and optimizer states from a file."""
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}.")
    return epoch



iou_metric = evaluate.load("mean_iou")
def mean_iou(preds, labels, num_classes):
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum()  # Fixed intersection calculation
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            iou = 0
        else:
            iou = float(intersection) / float(union)
        iou_list.append(iou)
    return np.mean(iou_list)


# Constants for normalization
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# Image transformations
image_transform = Compose([
    Resize((448, 448)),  # Resize all images to 448x448
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mask transformations
mask_transform = Compose([
    Resize((448, 448), interpolation=Image.NEAREST),  # Use nearest neighbor interpolation
    ToTensor()
])

class PlantDataset(Dataset):
    def __init__(self, samples, image_transform=None, mask_transform=None):
        self.samples = samples
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # Apply transformations to the mask
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        
        mask = mask.long() if not isinstance(mask, torch.LongTensor) else mask
        
        return image, mask
    
def load_datasets(root_dir, image_transform, mask_transform, split_ratio=0.8):
    image_dir = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks')

    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    samples = []
    
    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file)  # Use the same filename for the mask
        
        if os.path.exists(mask_path):  # Check if the mask file exists
            samples.append((img_path, mask_path))
        else:
            print(f"No corresponding mask found for image {image_file}")

    # Shuffle samples to ensure random distribution for training and validation
    split_point = int(len(samples) * split_ratio)
    train_samples = samples[:split_point]
    val_samples = samples[split_point:]


    # Create dataset instances with the correct arguments
    train_dataset = PlantDataset(train_samples, image_transform=image_transform, mask_transform=mask_transform)
    val_dataset = PlantDataset(val_samples, image_transform=image_transform, mask_transform=mask_transform)
    
    return train_dataset, val_dataset



# Load and split datasets
root_dir = 'C:\\Users\\Stell\\Desktop\\DVA 309\\Plant segmentation'
train_dataset, val_dataset = load_datasets(root_dir, image_transform, mask_transform)
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False)

def check_data_samples(dataloader):
    images, masks = next(iter(dataloader))
    figure, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for i in range(5):
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        image = (image - image.min()) / (image.max() - image.min())
        ax[0, i].imshow(image)
        ax[0, i].axis('off')

        # Make sure mask is 2D
        mask = masks[i].cpu().squeeze()
        if mask.ndim > 2:
            mask = mask[0]  # Take the first channel if mask has an extra dimension
        ax[1, i].imshow(mask, cmap='gray')
        ax[1, i].axis('off')
    plt.tight_layout()
    plt.show()



check_data_samples(train_dataloader)

def check_mask_values(dataloader):
    images, masks = next(iter(dataloader))
    unique_values = [torch.unique(mask).tolist() for mask in masks]
    print(f"Unique mask values: {unique_values}")

check_mask_values(train_dataloader)

def inspect_single_mask(dataloader):
    images, masks = next(iter(dataloader))
    mask = masks[0]  # Get the first mask
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.colorbar()
    plt.title('Single Mask Inspection')
    plt.show()
    
    # Print out unique values and their counts
    unique, counts = torch.unique(mask, return_counts=True)
    print(f"Unique values in the mask: {unique}")
    print(f"Counts for each value: {counts}")

inspect_single_mask(train_dataloader)


# Model setup
class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = torch.nn.Conv2d(config.hidden_size, 2, kernel_size=2)

    def forward(self, pixel_values, labels=None):
        outputs = self.dinov2(pixel_values, return_dict=True)
        features = outputs.last_hidden_state[:, 1:, :]
        features = features.permute(0, 2, 1).view(features.shape[0], -1, 32, 32)
        logits = self.classifier(features)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False)

        loss = None
        if labels is not None:
            labels = labels.squeeze(1)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SemanticSegmenterOutput(loss=loss, logits=logits)

model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-large")
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

train_losses = []
val_losses = []

# Training and validation loops
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_loss = 0
    num_batches = 0
    for images, masks in tqdm(train_dataloader):
        images, masks = images.to(model.device), masks.to(model.device)
        optimizer.zero_grad()
        outputs = model(images, labels=masks)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_train_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = total_train_loss / num_batches
    train_losses.append(avg_train_loss)

        
    preds = outputs.logits.argmax(dim=1)
    iou_metric.add_batch(predictions=preds, references=masks)

  
    num_classes = 2  
    ignore_index = -1 
    train_iou = iou_metric.compute(num_labels=num_classes, ignore_index=ignore_index)

    scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(train_dataloader)}, Train IoU: {train_iou["mean_iou"]}')

    save_checkpoint(model, optimizer, scheduler, epoch, file_path=f"checkpoint_oterdata_epoch_{epoch}.pth")

    # Validation phase
    model.eval()
    total_val_loss = 0
    num_batches = 0
    val_loss = 0
    iou_metric = evaluate.load("mean_iou", num_labels=num_classes, ignore_index=ignore_index)
    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(model.device), masks.to(model.device)
            outputs = model(images, labels=masks)
            val_loss += outputs.loss.item()
            total_val_loss += val_loss
            num_batches += 1
            # Update IoU metric
            preds = outputs.logits.argmax(dim=1)
            iou_metric.add_batch(predictions=preds, references=masks)
        
        avg_val_loss = total_val_loss / num_batches
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    val_iou = iou_metric.compute(num_labels = 2, ignore_index = -1)
    print(f'Validation Loss: {val_loss / len(val_dataloader)}, Validation IoU: {val_iou["mean_iou"]}')


def plot_loss_curve(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss_curve(train_losses, val_losses)


def visualize_batch(dataloader, num_images=5):
   
    images, masks = next(iter(dataloader))

   
    images_np = images.numpy().transpose((0, 2, 3, 1))
    masks_np = masks.numpy()

    
    if masks_np.ndim == 3:  # Shape is (batch_size, height, width)
        masks_np = masks_np[:, :, :, None] 

    # Normalize image for display
    images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())

   
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, figure=fig)

   
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.hstack(images_np))
    ax1.set_title('Images', fontsize=15)
    ax1.axis('off')

   
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(np.hstack(masks_np.squeeze()), cmap='gray')
    ax2.set_title('Masks', fontsize=15)
    ax2.axis('off')

    
    ax3 = fig.add_subplot(gs[2, 0])
    for i in range(num_images):
        ax3.imshow(images_np[i])
        ax3.imshow(masks_np[i].squeeze(), alpha=0.4, cmap='jet')  # Adjust alpha for mask transparency
    ax3.set_title('Overlay', fontsize=15)
    ax3.axis('off')

    plt.show()




visualize_batch(train_dataloader, num_images=5)


