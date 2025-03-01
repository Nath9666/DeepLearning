import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.cluster.vq import kmeans, vq

# Définition des chemins d'accès
train_path = "./Intel_Images/seg_train"
test_path = "./Intel_Images/seg_test"

# Fonction pour lister tous les fichiers dans un répertoire
def img_list(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

# Chargement des images et des étiquettes
def load_images_and_labels(path):
    class_names = os.listdir(path)
    image_paths = []
    image_classes = []
    for class_name in class_names:
        class_path = os.path.join(path, class_name)
        class_image_paths = img_list(class_path)
        image_paths += class_image_paths
        image_classes += [class_name] * len(class_image_paths)
    return image_paths, image_classes

# Chargement des données d'entraînement et de test
train_image_paths, train_image_classes = load_images_and_labels(train_path)
test_image_paths, test_image_classes = load_images_and_labels(test_path)

# Extraction des descripteurs ORB
orb = cv2.ORB_create()
des_list = []

for image_path in train_image_paths:
    image = cv2.imread(image_path, 0)
    kp, des = orb.detectAndCompute(image, None)
    if des is not None and des.shape[1] == 32:  # Vérifier que les descripteurs ont la dimension attendue
        des_list.append((image_path, des))

# Vérifier que des_list n'est pas vide
if not des_list:
    raise ValueError("Aucun descripteur ORB valide n'a été trouvé dans les images d'entraînement.")

# Création du vocabulaire avec KMeans
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    if descriptor.shape[1] == 32:  # Vérifier que les descripteurs ont la dimension attendue
        descriptors = np.vstack((descriptors, descriptor))

k = 200
voc, variance = kmeans(descriptors.astype(float), k, 1)

# Création de l'histogramme des caractéristiques
im_features = np.zeros((len(des_list), k), "float32")
for i in range(len(des_list)):
    if des_list[i][1].shape[1] == 32:  # Vérifier que les descripteurs ont la dimension attendue
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

# Standardisation des caractéristiques
stdslr = StandardScaler().fit(im_features)
im_features = stdslr.transform(im_features)

# Création du modèle de classification avec SVM
clf = SVC(kernel='linear')
clf.fit(im_features, train_image_classes[:len(des_list)])  # Ajuster la taille des classes d'entraînement

# Extraction des descripteurs ORB pour les images de test
des_list_test = []
for image_path in test_image_paths:
    image = cv2.imread(image_path, 0)
    kp, des = orb.detectAndCompute(image, None)
    if des is not None and des.shape[1] == 32:  # Vérifier que les descripteurs ont la dimension attendue
        des_list_test.append((image_path, des))

# Vérifier que des_list_test n'est pas vide
if not des_list_test:
    raise ValueError("Aucun descripteur ORB valide n'a été trouvé dans les images de test.")

# Création de l'histogramme des caractéristiques pour les images de test
test_features = np.zeros((len(des_list_test), k), "float32")
for i in range(len(des_list_test)):
    if des_list_test[i][1].shape[1] == 32:  # Vérifier que les descripteurs ont la dimension attendue
        words, distance = vq(des_list_test[i][1], voc)
        for w in words:
            test_features[i][w] += 1

# Standardisation des caractéristiques de test
test_features = stdslr.transform(test_features)

# Prédiction et évaluation du modèle
y_pred = clf.predict(test_features)
accuracy = accuracy_score(test_image_classes[:len(des_list_test)], y_pred)  # Ajuster la taille des classes de test
print(f"Test Accuracy: {accuracy * 100:.2f}%")