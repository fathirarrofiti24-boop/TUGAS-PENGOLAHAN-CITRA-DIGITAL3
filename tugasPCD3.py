import imageio.v3 as iio
import numpy as np 
import matplotlib.pyplot as plt

# 1.  membaca gambar berwarna 
gambar = iio.imread('gambar.png')

#2. mengambil channel RGB
rgb = gambar [:,:,:3]
gray = np.round(
    0.299 * rgb[:,:,0] +
    0.587 * rgb[:,:,1] +
    0.114 * rgb[:,:,2]
).astype(np.uint8)

# 4. menghitung jumlah piksel tiap intensitas 0-255
histogram = np.bincount(gray.ravel(), minlength=256)

# 5. menampilkan jumlah piksel setiap intensitas 
for intensitas in range(256):
    print(f"Intensitas {intensitas}: {histogram[intensitas]} piksel")

# 6. mencari intensitas yang paling dominan 
dominan = np.argmax(histogram)
print("\nIntesitas dominan:", dominan)
print('Jumlah piksel:", histogram[dominan])')

# 7. menampilkan gambar grayscale dan histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.title('Gambar Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(256), histogram, width=1 )
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas (0-255)")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()