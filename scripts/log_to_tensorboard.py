# log_to_tensorboard.py
import re
from tensorboardX import SummaryWriter
import time
import sys
import math

def parse_logs(log_file):
    writer = SummaryWriter()
    epoch = 0
    total_epochs = 0
    best_loss = math.inf
    lr_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse epoch progress
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                writer.add_scalar('00_Progress/Epoch', epoch, epoch)
                writer.add_scalar('00_Progress/Percentage', 
                                (epoch/total_epochs)*100, epoch)

            # Detailed loss parsing
            loss_match = re.search(
                r'loss: ([\d\.]+) - ' +
                r'regression_loss: ([\d\.]+) - ' +
                r'classification_loss: ([\d\.]+) - ' +
                r'lr: ([\d\.e-]+)', 
                line
            )
            
            if loss_match:
                total_loss = float(loss_match.group(1))
                reg_loss = float(loss_match.group(2))
                cls_loss = float(loss_match.group(3))
                lr = float(loss_match.group(4))
                
                # Core metrics
                writer.add_scalar('01_Loss/Total', total_loss, epoch)
                writer.add_scalar('01_Loss/Regression', reg_loss, epoch)
                writer.add_scalar('01_Loss/Classification', cls_loss, epoch)
                writer.add_scalar('02_Learning Rate', lr, epoch)
                
                # Advanced metrics
                writer.add_scalar('03_Loss Ratios/Regression vs Class', 
                                reg_loss/cls_loss if cls_loss !=0 else 0, epoch)
                writer.add_scalar('03_Loss Ratios/Current vs Initial', 
                                total_loss/(float(loss_match.group(1)) if epoch > 1 else 1, epoch))
                
                # Best loss tracking
                if total_loss < best_loss:
                    best_loss = total_loss
                writer.add_scalar('01_Loss/Best', best_loss, epoch)
                
                # LR tracking
                lr_values.append(lr)
                writer.add_histogram('02_LR Distribution', 
                                    np.array(lr_values), epoch)

            # Add custom text logging
            if 'ReduceLROnPlateau' in line:
                writer.add_text('Events/LR Change', line.strip(), epoch)
                
            if 'saving model' in line:
                writer.add_text('Events/Checkpoints', line.strip(), epoch)

    # Add hyperparameters
    writer.add_hparams(
        {'total_epochs': total_epochs},
        {'hparam/final_loss': total_loss}
    )
    
    writer.close()

if __name__ == "__main__":
    parse_logs(sys.argv[1])