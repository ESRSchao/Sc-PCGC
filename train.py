#!/usr/bin/env python3
"""Training script for Sc-PCGC."""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sc_pcgc.config import Config, load_config
from sc_pcgc.data import SemanticKITTIDataset
from sc_pcgc.compression import PointCloudCompressor
from sc_pcgc.utils import compute_batch_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Sc-PCGC model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/SemanticKITTI',
        help='Path to SemanticKITTI dataset'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for tensorboard logs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    writer,
    lambda_rate=0.01,
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_distortion = 0
    total_rate = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        points = batch['points'].to(device)
        labels = batch['labels'].to(device)
        
        # Compute priorities from labels using the preprocessing module
        from sc_pcgc.data.preprocessing import compute_semantic_priorities
        priorities = torch.zeros_like(labels, dtype=torch.float32)
        for b in range(labels.shape[0]):
            labels_np = labels[b].cpu().numpy()
            priorities_np = compute_semantic_priorities(labels_np)
            priorities[b] = torch.from_numpy(priorities_np).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, encoded = model(points, priorities)
        
        # Compute loss
        loss, metrics = model.compute_loss(
            points, reconstructed, encoded, lambda_rate
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += metrics['loss'].item()
        total_distortion += metrics['distortion'].item()
        total_rate += metrics['rate'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss'].item():.4f}",
            'dist': f"{metrics['distortion'].item():.4f}",
            'rate': f"{metrics['rate'].item():.2f}",
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('train/loss', metrics['loss'].item(), global_step)
        writer.add_scalar('train/distortion', metrics['distortion'].item(), global_step)
        writer.add_scalar('train/rate', metrics['rate'].item(), global_step)
    
    avg_loss = total_loss / len(dataloader)
    avg_distortion = total_distortion / len(dataloader)
    avg_rate = total_rate / len(dataloader)
    
    return avg_loss, avg_distortion, avg_rate


def validate(model, dataloader, device, epoch, writer):
    """Validate the model."""
    model.eval()
    
    total_metrics = {
        'chamfer_distance': 0,
        'psnr': 0,
        'compression_ratio': 0,
    }
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            
            # Compute priorities from labels using the preprocessing module
            from sc_pcgc.data.preprocessing import compute_semantic_priorities
            priorities = torch.zeros_like(labels, dtype=torch.float32)
            for b in range(labels.shape[0]):
                labels_np = labels[b].cpu().numpy()
                priorities_np = compute_semantic_priorities(labels_np)
                priorities[b] = torch.from_numpy(priorities_np).to(device)
            
            # Forward pass
            reconstructed, encoded = model(points, priorities)
            
            # Compute metrics
            metrics = compute_batch_metrics(points, reconstructed, encoded)
            
            for key, value in metrics.items():
                if key in total_metrics:
                    total_metrics[key] += value
            
            num_batches += 1
    
    # Average metrics
    avg_metrics = {
        key: value / num_batches if num_batches > 0 else 0
        for key, value in total_metrics.items()
    }
    
    # Log to tensorboard
    for key, value in avg_metrics.items():
        writer.add_scalar(f'val/{key}', value, epoch)
    
    return avg_metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.data_root:
        config.data.data_root = args.data_root
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SemanticKITTIDataset(
        data_root=config.data.data_root,
        split='train',
    )
    val_dataset = SemanticKITTIDataset(
        data_root=config.data.data_root,
        split='val',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = PointCloudCompressor(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_depth=config.octree.max_depth,
        voxel_size=config.octree.voxel_size,
        quantization_bits=config.compression.quantization_bits,
        use_entropy_coding=config.compression.use_entropy_coding,
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        
        # Train
        train_loss, train_dist, train_rate = train_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Distortion: {train_dist:.4f}, Rate: {train_rate:.2f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch, writer)
        print(f"Val - Metrics: {val_metrics}")
        
        # Learning rate scheduling
        scheduler.step(train_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'config': config.to_dict(),
        }, checkpoint_path)
        
        # Save best model
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': config.to_dict(),
            }, best_path)
            print(f"Saved best model to {best_path}")
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
