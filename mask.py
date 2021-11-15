import torch
import cv2

patch_num = 4

img = cv2.imread("img1.jpg")
img = cv2.resize(img, (512, 512))
img = torch.from_numpy(img)
img = img.unsqueeze(0).permute((0, 3, 1, 2))
print(img.size())

img = img.flatten(2).transpose(1, 2)
print(img.size())

tmp = img[0].view(2, 256, 2, 256, 3)[1, :, 1, :, :]
tmp = tmp.numpy()
cv2.imwrite("tmp.jpg", tmp)

B, N, C = img.size()
img = img.view(B, patch_num, N // patch_num, C)
print(img.size())

patch = img[0, 1, :, :]
print(patch.size())

patch = patch.view(512 // 2, 512 // 2, C)
print(patch.size())

# patch = patch.numpy()
# print(patch.shape)

# cv2.imwrite("patch.jpg", patch)

a = torch.randperm(5)
print(a)

x = torch.tensor([10, 20, 30, 40, 50])
rand_x = x[a]
print(rand_x)

unrand_x = rand_x[a.sort()[1]]
print(unrand_x)
