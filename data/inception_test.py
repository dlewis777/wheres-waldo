import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

inception = models.inception_v3(pretrained=True)

class WaldoDataset(Dataset):
