from sklearn import datasets
from matplotlib import pyplot as plt

lfw = datasets.fetch_lfw_people(min_faces_per_person=10, resize=0.4)

plt.figure(figsize=(15,3))

for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(lfw.image[i], cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])
    
plt.show()