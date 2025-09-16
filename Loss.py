import torch
import torch.nn as nn
import torch.nn.functional as F
from piqa import SSIM, LPIPS
from typing import Optional, Dict

class NoiseAwareCompositeLoss(nn.Module):
    """
    Noise-Aware Composite Loss Function for RAW image reconstruction.
    
    This loss combines reconstruction (L1 + Hard Logarithmic), structural (SSIM), 
    and perceptual (LPIPS) losses with configurable weights.
    
    Args:
        lambda_rec: Weight for reconstruction loss (default: 1.0)
        lambda_str: Weight for structural loss (default: 0.1)
        lambda_perc: Weight for perceptual loss (default: 0.1)
        lambda_hlog: Weight for hard logarithmic loss (default: 0.05)
        eps: Small value for numerical stability (default: 1e-8)
        device: Device to run the loss computation on
    """
    
    def __init__(self, 
                 lambda_rec: float = 1.0,
                 lambda_str: float = 0.1, 
                 lambda_perc: float = 0.1,
                 lambda_hlog: float = 0.05,
                 eps: float = 1e-8,
                 device: Optional[torch.device] = None):
        super().__init__()
        
        self.lambda_rec = lambda_rec
        self.lambda_str = lambda_str
        self.lambda_perc = lambda_perc
        self.lambda_hlog = lambda_hlog
        self.eps = eps
        
        # Initialize device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        
        # SSIM with appropriate parameters for image reconstruction
        self.ssim = SSIM(n_channels=1, value_range=1.0).to(self.device)
        
        # LPIPS for perceptual similarity
        self.lpips = LPIPS(network='vgg', pretrained=True).to(self.device)
        
        print(f"NoiseAwareCompositeLoss initialized with weights: "
              f"rec={lambda_rec}, str={lambda_str}, perc={lambda_perc}, hlog={lambda_hlog}")
    
    def hard_logarithmic_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Hard Logarithmic Loss that heavily penalizes large errors.
        
        Args:
            pred: Predicted tensor of shape (N, C, H, W)
            target: Target tensor of shape (N, C, H, W)
            
        Returns:
            Hard logarithmic loss value
        """
        # Calculate absolute differences
        abs_diff = torch.abs(pred - target)
        
        # Clip differences to [0, 1] range as in the paper
        clipped_diff = torch.clamp(abs_diff, max=1.0)
        
        # Compute logarithmic term with numerical stability
        log_term = -torch.log(1.0 - clipped_diff + self.eps)
        
        # Average over all dimensions
        return torch.mean(log_term)
    
    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined reconstruction loss (L1 + Hard Logarithmic).
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Combined reconstruction loss value
        """
        l1_loss = self.l1_loss(pred, target)
        hlog_loss = self.hard_logarithmic_loss(pred, target)
        
        return l1_loss + self.lambda_hlog * hlog_loss
    
    def structural_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute structural loss using SSIM.
        
        Args:
            pred: Predicted tensor of shape (N, C, H, W)
            target: Target tensor of shape (N, C, H, W)
            
        Returns:
            Structural loss value (1 - SSIM)
        """
        # SSIM returns similarity, so we convert to loss: 1 - SSIM
        ssim_value = self.ssim(pred, target)
        return 1.0 - ssim_value
    
    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using LPIPS.
        
        Args:
            pred: Predicted tensor of shape (N, C, H, W)
            target: Target tensor of shape (N, C, H, W)
            
        Returns:
            Perceptual loss value
        """
        return self.lpips(pred, target)
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Forward pass to compute the total composite loss.
        
        Args:
            pred: Predicted tensor of shape (N, C, H, W)
            target: Target tensor of shape (N, C, H, W)
            return_components: If True, return individual loss components
            
        Returns:
            Total loss value or dictionary of loss components
        """
        # Ensure tensors are on the correct device
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # Compute individual loss components
        rec_loss = self.reconstruction_loss(pred, target)
        str_loss = self.structural_loss(pred, target)
        perc_loss = self.perceptual_loss(pred, target)
        
        # Combine losses with weights
        total_loss = (self.lambda_rec * rec_loss + 
                     self.lambda_str * str_loss + 
                     self.lambda_perc * perc_loss)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'reconstruction_loss': rec_loss,
                'structural_loss': str_loss,
                'perceptual_loss': perc_loss,
                'l1_loss': self.l1_loss(pred, target),
                'hlog_loss': self.hard_logarithmic_loss(pred, target)
            }
        else:
            return total_loss
    
    def to(self, device: torch.device):
        """Move all components to the specified device."""
        super().to(device)
        self.device = device
        self.ssim = self.ssim.to(device)
        self.lpips = self.lpips.to(device)
        return self


# Example usage and test function
def test_noise_aware_loss():
    """Test function to demonstrate the usage of NoiseAwareCompositeLoss."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample data (batch_size=2, channels=3, height=64, width=64)
    batch_size, channels, height, width = 2, 3, 64, 64
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    
    # Initialize the loss function
    loss_fn = NoiseAwareCompositeLoss()
    
    # Compute total loss
    total_loss = loss_fn(pred, target)
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Compute loss with individual components
    loss_components = loss_fn(pred, target, return_components=True)
    
    print("\nLoss components:")
    for name, value in loss_components.items():
        print(f"{name}: {value.item():.6f}")
    
    return loss_components

