import random

def train_test_split(corpus, test_size=0.25):
    train = list(corpus)
    n_test = int(len(train) * test_size)
    indices = random.sample(range(len(train)), n_test)
    test = [train.pop(i) for i in sorted(indices)]
    return train, test



if __name__ == '__main__':
    x = list(range(10))

    print(train_test_split(x, 0.3))
    print(x)