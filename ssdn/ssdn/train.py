import os
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
from itertools import chain

from ssdn.models.noise2noise import Noise2Noise

bsds300_path = datasets_path + 'BSDS300/'
bsds300_gaussian_train_path = bsds300_path + 'images/gaussian_train/'
bsds300_gaussian_test_path = bsds300_path + 'images/gaussian_test/'
bsds300_train_path = bsds300_path + 'images/train/'
bsds300_test_path = bsds300_path + 'images/test/'

def train_noise2noise():
    """Train the Noise2Noise model
    TODO docs innit
    """
    transform_to_tensor = torchvision.transforms.ToTensor()

    gaussian_trainset = []
    for r, d, f in os.walk(bsds300_gaussian_train_path):
        for file in f:
            file_path = os.path.join(r, file)
            image = Image.open(file_path)
            image_tensor = transform_to_tensor(image)
            if image_tensor.shape[1] > 321:
                image_tensor = image_tensor.permute(0, 2, 1)
            gaussian_trainset.append(image_tensor)
    target_trainset = []
    for r, d, f in os.walk(bsds300_train_path):
        for file in f:
            file_path = os.path.join(r, file)
            image = Image.open(file_path)
            image_tensor = transform_to_tensor(image)
            if image_tensor.shape[1] > 321:
                image_tensor = image_tensor.permute(0, 2, 1)
            target_trainset.append(image_tensor)
    trainset = list(zip(gaussian_trainset, target_trainset))

    gaussian_testset = []
    for r, d, f in os.walk(bsds300_gaussian_test_path):
        for file in f:
            file_path = os.path.join(r, file)
            image = Image.open(file_path)
            image_tensor = transform_to_tensor(image)
            if image_tensor.shape[1] > 321:
                image_tensor = image_tensor.permute(0, 2, 1)
            gaussian_testset.append(image_tensor)
    target_testset = []
    for r, d, f in os.walk(bsds300_test_path):
        for file in f:
            file_path = os.path.join(r, file)
            image = Image.open(file_path)
            image_tensor = transform_to_tensor(image)
            if image_tensor.shape[1] > 321:
                image_tensor = image_tensor.permute(0, 2, 1)
            target_testset.append(image_tensor)
    testset = list(zip(gaussian_testset, target_testset))

    # fix random seed for reproducibility
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=True)

    # build the model
    model = Noise2Noise()

    # define the loss function and the optimiser
    # TODO need to use signal-to-noise ratio somewhere?
    loss_function = nn.MSELoss()
    optimiser = optim.Adam(model.parameters())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
    trial.with_generators(trainloader, test_generator=testloader)
    trial.run(epochs=1)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)


if __name__ == "__main__":
    train_noise2noise()
