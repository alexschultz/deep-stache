import cv2

stache = cv2.imread("stache.png", -1)
face = cv2.imread("face.jpeg", -1)
resizedStache = cv2.resize(stache, (face.shape[1], face.shape[0]))
x_offset = y_offset = 0

y1, y2 = y_offset, y_offset + resizedStache.shape[0]
x1, x2 = x_offset, x_offset + resizedStache.shape[1]

alpha_s = resizedStache[:, :, 3] / 255.0
alpha_l = 1.0 - alpha_s

for c in range(0, 3):
    face[y1:y2, x1:x2, c] = (alpha_s * resizedStache[:, :, c] + alpha_l * face[y1:y2, x1:x2, c])

cv2.imshow('face with mustache', face)

cv2.waitKey(0)
cv2.destroyAllWindows()
