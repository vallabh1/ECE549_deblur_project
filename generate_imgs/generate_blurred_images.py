from motion_blur import MotionBlurGenerator
import cv2

img = cv2.imread("./squirrel.jpg")

blur_gen = MotionBlurGenerator()

# 1. Linear blur
blur1 = blur_gen.apply_blur(img, mode="linear", length=30, angle=45)

# 2. Camera shake blur
blur2 = blur_gen.apply_blur(img, mode="shake", steps=12, max_length=8)

# 3. Angular blur
blur3 = blur_gen.apply_blur(img, mode="angular", step=15, angle_range=10)

cv2.imwrite("./blur_linear.jpg", blur1)
cv2.imwrite("./blur_shake.jpg", blur2)
cv2.imwrite("./blur_angular.jpg", blur3)
