def get_mean_and_std(dataset):
    if dataset == "cifar10":
        means = [0.49139968, 0.48215841, 0.44653091]
        stds = [0.24703223, 0.24348513, 0.26158784]
    elif dataset == "svhn":
        means = [0.4376821, 0.4437697, 0.47280442]
        stds = [0.19803012, 0.20101562, 0.19703614]
    elif dataset == "cifar100":
        means = [0.50736203, 0.48668956, 0.44108857],
        stds = [0.26748815, 0.2565931,  0.27630851]
    else:
        assert False
    return means, stds


def normalize_images(images, dataset):
    images = images / 255.0
    mean, std = get_mean_and_std(dataset)
    images = (images - mean) / std
    return images
