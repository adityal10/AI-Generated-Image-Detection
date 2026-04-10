import torch
import torch.nn as nn
import glob
import pickle
import tqdm
from torch.utils.data import DataLoader, random_split

from models import load_dinvo3, DinoDetector_V3
from data_loaders import SimpleTrainingDataset
from compute_metrics import compute_accuracy, compute_ap



def training_loop(REPO_DIR, WEIGHT_DIR, DEVICE, real_image_paths, final_fake_image_paths, ):
    dinov3_vitb16 = load_dinvo3(REPO_DIR, WEIGHT_DIR, DEVICE)

    model = DinoDetector_V3(dinov3_vitb16)
    model = model.to(DEVICE)


    dataset = SimpleTrainingDataset(real_image_paths, final_fake_image_paths)
    batch_size = 8
    
    # split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train: {train_size} images | Val: {val_size} images")

    # Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-7, momentum=0.9) #5e-7 given in the paper
    num_epochs = 100
    

    for epoch in range(num_epochs):
        model.train()
        model.to(DEVICE)
        
        train_loss = 0.0
        train_acc = 0.0
        train_ap = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += compute_accuracy(outputs, labels)
            train_ap += compute_ap(outputs, labels)
            num_batches += 1
            
            pbar.set_postfix({
                'loss': train_loss / num_batches,
                'acc': train_acc / num_batches,
                'ap': train_ap / num_batches
            })
        
        train_loss /= num_batches
        train_acc /= num_batches
        train_ap /= num_batches
        
        # Validation to detect overfitting)
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_ap = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += compute_accuracy(outputs, labels)
                val_ap += compute_ap(outputs, labels)
                val_batches += 1
        
        val_loss /= val_batches
        val_acc /= val_batches
        val_ap /= val_batches
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AP: {train_ap:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AP: {val_ap:.4f}")