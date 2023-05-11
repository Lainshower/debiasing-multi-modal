# 감사링(from 진수)

### 대부분의 Code는 https://github.com/kohpangwei/group_DRO를 참고하는듯?
### CelebA Dataset source: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset (Kaggle)

### img_align_celeba.zip: All the face images, cropped and aligned 
- original이랑 crop/aligned version이 있는데 grop_DRO랑 최근 떨어진 논문 둘다 crop/aligned IMG 씀
### list_eval_partition.csv: Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
### list_attr_celeba.csv: Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative

-> Blond_Hair랑 Male이 spurious correlation detection에서 대표적으로 검증하는 Attribute인가..? : 맞는듯 (contrastive adapter & bst 논문 둘다 이렇게 실험 setting 함)
--> Class {blond <> non-blond}
--> Group(=spurious cofounder) {male <> female}

-> 아래 2개는 안씀
### list_bbox_celeba.csv: Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
### list_landmarks_align_celeba.csv: Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth