from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, shutil

def train_test_split(dataroot, split_ratio, train_transform, test_transform):
    # subdirs 리스트에는 class가 저장되어 있음. (ex. ['bicycle', 'car'])
    subdirs = os.listdir(dataroot)

    # train / test 디렉토리 생성
    trainroot = os.path.join(dataroot, 'train')
    testroot = os.path.join(dataroot, 'test')

    if os.path.exists(trainroot) and os.path.exists(testroot):
        pass
    else:
        if not os.path.exists(trainroot):
            os.makedirs(trainroot)
        if not os.path.exists(testroot):
            os.makedirs(testroot)

        # train / test 분리 작업
        for subdir in subdirs:
            full_path = os.path.join(dataroot, subdir)
            files = os.listdir(full_path)

            # train set과 test set으로 쓰일 파일들을 분리
            train_idx = int(len(files)*split_ratio)
            train_files = [os.path.join(full_path, f) for f in files[:train_idx]]
            test_files = [os.path.join(full_path, f) for f in files[train_idx:]]

            # train/bicycle, train/car, test/bicycle, test/car 폴더 생성
            os.makedirs(os.path.join(trainroot, subdir), exist_ok=True)
            os.makedirs(os.path.join(testroot, subdir), exist_ok=True)

            # train directory로 이미지 파일 복사
            for trf in train_files:
                shutil.copy(trf, os.path.join(trainroot, subdir))

            # test directory로 이미지 파일 복사
            for tsf in test_files:
                shutil.copy(tsf, os.path.join(testroot, subdir))

    # 이렇게 생성된 train directory, test directory를 dataroot로써 각각 dataset 생성
    train_set = ImageFolder(root=trainroot, transform=train_transform)
    test_set = ImageFolder(root=testroot, transform=test_transform)

    return train_set, test_set

def get_loader(dataroot, split_ratio, batch_size, shuffle, train_transform, test_transform, num_workers):
    train_set, test_set = train_test_split(dataroot, split_ratio, train_transform, test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader