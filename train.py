import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from datetime import datetime
import argparse

from data_loader import get_dataloaders
from model import SegmentationModel, get_available_models, get_available_encoders
from utils import Visualizer, MetricsCalculator, save_checkpoint

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.writer = SummaryWriter(f"runs/{self.timestamp}")
        
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size=config.batch_size,
            img_size=config.img_size
        )
        
        self.models_to_compare = config.models if config.compare_models else [config.model_name]
        
        self.results = {}
        
    def train_single_model(self, model_name, encoder_name):
        """آموزش یک مدل خاص"""
        print(f"\n{'='*50}")
        print(f"Training {model_name} with {encoder_name} encoder")
        print(f"{'='*50}")
        
        model_wrapper = SegmentationModel(
            model_name=model_name,
            encoder_name=encoder_name,
            num_classes=5
        )
        model = model_wrapper.to_device(self.device).get_model()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        train_losses = []
        val_metrics_history = []
        best_miou = 0.0
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            
            for images, masks, _ in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_loss)
            
            val_metrics = self.validate_model(model, self.val_loader)
            val_metrics_history.append(val_metrics)
            
            scheduler.step(avg_loss)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {avg_loss:.4f}")
            print(f"Val mIoU: {val_metrics['mIoU']:.4f}, Accuracy: {val_metrics['Accuracy']:.4f}")
            print(f"Val F1: {val_metrics['F1-Score']:.4f}")
            print("-" * 30)
            
            if val_metrics['mIoU'] > best_miou:
                best_miou = val_metrics['mIoU']
                checkpoint_path = os.path.join(
                    self.output_dir, 
                    f"best_{model_name}_{encoder_name}.pth"
                )
                save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
            
            self.writer.add_scalar(f'{model_name}_{encoder_name}/Train_Loss', avg_loss, epoch)
            self.writer.add_scalar(f'{model_name}_{encoder_name}/Val_mIoU', val_metrics['mIoU'], epoch)
            self.writer.add_scalar(f'{model_name}_{encoder_name}/Val_Accuracy', val_metrics['Accuracy'], epoch)
        
        return {
            'model_name': model_name,
            'encoder_name': encoder_name,
            'train_losses': train_losses,
            'val_metrics': val_metrics_history,
            'best_miou': best_miou,
            'model': model
        }
    
    def validate_model(self, model, dataloader):
        """ارزیابی مدل روی دیتالودر"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks, _ in dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = MetricsCalculator.calculate_metrics(all_preds, all_targets)
        return metrics
    
    def visualize_results(self, model, dataloader, num_samples=3):
        """ویژوالایزیشن نتایج"""
        model.eval()
        images_list, true_masks_list, pred_masks_list, filenames_list = [], [], [], []
        
        with torch.no_grad():
            for i, (images, masks, filenames) in enumerate(dataloader):
                if i >= 1:
                    break
                
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                images_list.extend(images.cpu())
                true_masks_list.extend(masks.cpu())
                pred_masks_list.extend(preds.cpu())
                filenames_list.extend(filenames)
        
        fig = Visualizer.create_comparison_grid(
            images_list[:num_samples],
            true_masks_list[:num_samples],
            pred_masks_list[:num_samples],
            filenames_list[:num_samples],
            num_samples
        )
        
        return fig
    
    def compare_models(self):
        """مقایسه چند مدل مختلف"""
        for model_name in self.models_to_compare:
            for encoder_name in self.config.encoders:
                result = self.train_single_model(model_name, encoder_name)
                model_key = f"{model_name}_{encoder_name}"
                self.results[model_key] = result
                
                fig = self.visualize_results(
                    result['model'], 
                    self.val_loader, 
                    num_samples=3
                )
                
                viz_path = os.path.join(self.output_dir, f"viz_{model_key}.png")
                fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved: {viz_path}")
        
        self.print_comparison_results()
        
    def print_comparison_results(self):
        """چاپ نتایج مقایسه مدل‌ها"""
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON RESULTS")
        print("="*60)
        
        for model_key, result in self.results.items():
            print(f"\n{model_key}:")
            print(f"  Best mIoU: {result['best_miou']:.4f}")
            print(f"  Final Accuracy: {result['val_metrics'][-1]['Accuracy']:.4f}")
            print(f"  Final F1-Score: {result['val_metrics'][-1]['F1-Score']:.4f}")
    
    def run(self):
        """اجرای اصلی آموزش"""
        start_time = time.time()
        
        if self.config.compare_models:
            self.compare_models()
        else:
            result = self.train_single_model(
                self.config.model_name, 
                self.config.encoder_name
            )
            self.results[self.config.model_name] = result
            
            fig = self.visualize_results(
                result['model'], 
                self.val_loader, 
                num_samples=3
            )
            viz_path = os.path.join(self.output_dir, f"viz_{self.config.model_name}.png")
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
        
        end_time = time.time()
        print(f"\nTotal training time: {(end_time - start_time)/60:.2f} minutes")
        print(f"Results saved in: {self.output_dir}")
        
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Satellite Image Segmentation Training')
    
    parser.add_argument('--model_name', type=str, default='unet', 
                       choices=get_available_models(), help='Model architecture')
    parser.add_argument('--encoder_name', type=str, default='resnet34',
                       choices=get_available_encoders(), help='Encoder backbone')
    parser.add_argument('--compare_models', action='store_true', 
                       help='Compare multiple models')
    parser.add_argument('--models', nargs='+', default=['unet', 'deeplabv3plus'],
                       choices=get_available_models(), help='Models to compare')
    parser.add_argument('--encoders', nargs='+', default=['resnet34', 'efficientnet-b0'],
                       choices=get_available_encoders(), help='Encoders to compare')
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    
    args = parser.parse_args()
    
    if not all(os.path.exists(f"{split}.csv") for split in ['train', 'val', 'test']):
        print("Creating CSV files...")
        from create_csv import create_csv_files
        create_csv_files()
    
    trainer = Trainer(args)
    trainer.run()

if __name__ == "__main__":
    main()