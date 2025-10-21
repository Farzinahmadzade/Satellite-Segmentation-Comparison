import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from datetime import datetime
import argparse
import logging
from data_loader import get_dataloaders
from model import SegmentationModel, get_available_models, get_available_encoders
from utils import Visualizer, MetricsCalculator, save_checkpoint

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
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
        logging.info(f"\n{'='*50}")
        logging.info(f"Training {model_name} with {encoder_name} encoder")
        logging.info(f"{'='*50}")
        
        model_wrapper = SegmentationModel(model_name=model_name, encoder_name=encoder_name, num_classes=5)
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
            
            logging.info(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_loss:.4f} | "
                         f"Val mIoU: {val_metrics['mIoU']:.4f} | Accuracy: {val_metrics['Accuracy']:.4f} | "
                         f"F1-Score: {val_metrics['F1-Score']:.4f}")
            
            self.writer.add_scalar(f'{model_name}_{encoder_name}/Train_Loss', avg_loss, epoch)
            self.writer.add_scalar(f'{model_name}_{encoder_name}/Val_mIoU', val_metrics['mIoU'], epoch)
            self.writer.add_scalar(f'{model_name}_{encoder_name}/Val_Accuracy', val_metrics['Accuracy'], epoch)
            
            if val_metrics['mIoU'] > best_miou:
                best_miou = val_metrics['mIoU']
                checkpoint_path = os.path.join(self.output_dir, f"best_{model_name}_{encoder_name}.pth")
                save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
                logging.info(f"New best model saved: {checkpoint_path}")
        
        return {
            'model_name': model_name,
            'encoder_name': encoder_name,
            'train_losses': train_losses,
            'val_metrics': val_metrics_history,
            'best_miou': best_miou,
            'model': model
        }

    def validate_model(self, model, dataloader):
        """Evaluate model on validation or test set."""
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
        """Visualize input, ground truth, prediction, and overlay."""
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
        """Train and compare multiple models."""
        for model_name in self.models_to_compare:
            for encoder_name in self.config.encoders:
                result = self.train_single_model(model_name, encoder_name)
                model_key = f"{model_name}_{encoder_name}"
                self.results[model_key] = result
                
                fig = self.visualize_results(result['model'], self.val_loader)
                viz_path = os.path.join(self.output_dir, f"viz_{model_key}.png")
                fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                logging.info(f"Visualization saved: {viz_path}")
        
        self.print_comparison_results()

    def print_comparison_results(self):
        logging.info("\n" + "="*60)
        logging.info("FINAL MODEL COMPARISON RESULTS")
        logging.info("="*60)
        for model_key, result in self.results.items():
            logging.info(f"{model_key} | Best mIoU: {result['best_miou']:.4f} | "
                         f"Final Accuracy: {result['val_metrics'][-1]['Accuracy']:.4f} | "
                         f"Final F1-Score: {result['val_metrics'][-1]['F1-Score']:.4f}")

    def run(self):
        """Main training loop."""
        start_time = time.time()
        if self.config.compare_models:
            self.compare_models()
        else:
            result = self.train_single_model(self.config.model_name, self.config.encoder_name)
            self.results[self.config.model_name] = result
            fig = self.visualize_results(result['model'], self.val_loader)
            viz_path = os.path.join(self.output_dir, f"viz_{self.config.model_name}.png")
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
            logging.info(f"Visualization saved: {viz_path}")
        
        end_time = time.time()
        logging.info(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
        logging.info(f"Results saved in: {self.output_dir}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Satellite Image Segmentation Training')
    
    parser.add_argument('--model_name', type=str, default='unet', choices=get_available_models())
    parser.add_argument('--encoder_name', type=str, default='resnet34', choices=get_available_encoders())
    parser.add_argument('--compare_models', action='store_true')
    parser.add_argument('--models', nargs='+', default=['unet', 'deeplabv3plus'], choices=get_available_models())
    parser.add_argument('--encoders', nargs='+', default=['resnet34', 'efficientnet-b0'], choices=get_available_encoders())
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.run()

if __name__ == "__main__":
    main()