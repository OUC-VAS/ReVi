import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc(gt, pred):
    """Calculate AUC score."""
    y_true = gt.flatten() / 255.0
    y_scores = pred.flatten() / 255.0
    y_true = (y_true > 0.5).astype(np.uint8) #Binarize GT to ignore rare, minor anomalies.

    # Compute AUC
    return roc_auc_score(y_true, y_scores)

def auc(folder1, folder2):
    """iterate over images in folders and compute AUC scores."""
    auc_scores = []
    zero_auc_filenames = []  # Store filenames with AUC score == 0
    a = 0

    for filename in os.listdir(folder1):
        if filename.endswith((".jpg", ".png", ".tif")):
            image_path1 = os.path.join(folder1, filename)

            base_name = os.path.splitext(filename)[0]
            possible_extensions = [".jpg", ".png", ".tif", ".jpeg"]
            image_path2 = None

            for ext in possible_extensions:
                test_path = os.path.join(folder2, base_name + ext)
                if os.path.exists(test_path):
                    image_path2 = test_path
                    break

            if image_path2 is not None:
                img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

                if img1 is None or img2 is None:
                    print(f"Cannot read image: {filename} or {os.path.basename(image_path2)}")
                    continue

                if img1.shape != img2.shape:
                    print(f"Image size mismatch: {filename}, {img1.shape} vs {os.path.basename(image_path2)}, {img2.shape}")
                    continue

                auc_score = calculate_auc(img1, img2)
                auc_scores.append(auc_score)
                a += 1
                print(f"AUC score for '{filename}' vs '{os.path.basename(image_path2)}': {auc_score:.4f}")

                if auc_score == 0.0:
                    zero_auc_filenames.append(filename)

    if auc_scores:
        average_auc = np.mean(auc_scores)
        # print(f"Average AUC score: {average_auc:.4f}")
        # print(f"Number of files with AUC score 0: {len(zero_auc_filenames)}")
        # print(a)

        with open('zero_auc_files.txt', 'w') as f:
            for file_name in zero_auc_filenames:
                f.write(file_name + '\n')
        #print(f"Filenames with AUC score 0 have been stored in 'zero_auc_files.txt'")
        return average_auc
    else:
        print("No images available for AUC score calculation.")

def binarize_image(image):
    """Binarize images"""
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

def calculate_f1_score(gt, pred):
    """Calculate F1 score"""
    TP = np.sum((gt == 255) & (pred == 255))
    FP = np.sum((gt == 0) & (pred == 255))
    FN = np.sum((gt == 255) & (pred == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def f1(folder1, folder2):
    """iterate over images in folders and compute F1 scores"""
    f1_scores = []
    zero_f1_filenames = []  # Store filenames with F1 score == 0
    b =0

    # 遍历第一个文件夹中的图像
    for filename in os.listdir(folder1):
        if filename.endswith((".jpg", ".png", ".tif")):
            image_path1 = os.path.join(folder1, filename)

            base_name = os.path.splitext(filename)[0]
            possible_extensions = [".jpg", ".png", ".tif", ".jpeg"]
            image_path2 = None

            for ext in possible_extensions:
                test_path = os.path.join(folder2, base_name + ext)
                if os.path.exists(test_path):
                    image_path2 = test_path
                    break

            if image_path2 is not None:
                img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

                if img1 is None or img2 is None:
                    print(f"Cannot read image: {filename} or {os.path.basename(image_path2)}")
                    continue

                if img1.shape != img2.shape:
                    print(f"Image size mismatch: {filename}, {img1.shape} vs {os.path.basename(image_path2)}, {img2.shape}")
                    continue

                binary_img1 = binarize_image(img1)
                binary_img2 = binarize_image(img2)

                f1_score = calculate_f1_score(binary_img1, binary_img2)
                f1_scores.append(f1_score)
                print(f"F1 score for '{filename}' vs '{os.path.basename(image_path2)}' : {f1_score:.4f}")
                #b = b+1

                if f1_score == 0:
                    zero_f1_filenames.append(filename)

    if f1_scores:
        average_f1_score = np.mean(f1_scores)
        #print(f"Average F1 score: {average_f1_score:.4f}")
        #print(b)

        # Write filenames with F1 score 0 into a text file
        with open('zero_f1_files.txt', 'w') as f:
            for file_name in zero_f1_filenames:
                f.write(file_name + '\n')
        return average_f1_score
    else:
        print("No images available for F1 score calculation.")

def caculate_metric(output_folder, gt_folder):
    average_auc = auc(output_folder, gt_folder)
    average_f1 = f1(output_folder, gt_folder)
    print(f"Average AUC score: {average_auc:.4f}")
    print(f"Average F1 score: {average_f1:.4f}")