import matplotlib.pyplot as plt
import numpy as np

imgs = np.load('food_images.npy')
idxs = np.random.permutation(len(imgs))
imgs = imgs[idxs]
ratings = []

print()

for im in imgs:
    plt.imshow(im)
    plt.show()
    rating = input('Enter rating (0-5): ')
    if rating == 'q' or rating == 'Q':
        break
    ratings.append(float(rating))
    plt.close()

output = np.zeros((2, len(ratings)))
output[0] = idxs[:len(ratings)]
output[1] = ratings
np.save('steve.npy', output)

