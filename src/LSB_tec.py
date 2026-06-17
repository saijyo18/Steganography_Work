#!/usr/bin/env python
# coding: utf-8

# ## To implement LSB technique from scratch

# In[ ]:


## LSB technique


# In[ ]:


from PIL import Image
import numpy as np

def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def encode_lsb(image_path, secret_message, output_path):

    img = Image.open(image_path)
    img_array = np.array(img)

    binary_message = text_to_binary(secret_message)

    # End marker
    binary_message += '1111111111111110'

    flat_pixels = img_array.flatten()

    if len(binary_message) > len(flat_pixels): ## condition
        raise ValueError("Message too large for image")

    for i, bit in enumerate(binary_message):

        flat_pixels[i] = (flat_pixels[i] & 0xFE) | int(bit)

    stego_array = flat_pixels.reshape(img_array.shape)

    stego_img = Image.fromarray(stego_array.astype(np.uint8))
    stego_img.save(output_path)

    return img_array, stego_array


# In[ ]:


secret="Maybe its the way you say my name Maybe its the way you play your game But its goodI ve never known anybody like youBut its so good, Ive never dreamed of nobody like youAnd Ive heard of a love that comes once in a lifetimeAnd Im pretty sure "
original, stego = encode_lsb(
    "original.png",
    secret,
    "LSB_stego.png"
)

print("Message embedded successfully")


# In[ ]:


from skimage.metrics import structural_similarity as ssim

def mse(original, stego):
    return np.mean((original.astype(float) - stego.astype(float))**2)

def psnr(original, stego):

    mse_val = mse(original, stego)

    if mse_val == 0:
        return float('inf')

    return 10 * np.log10((255**2) / mse_val)


def calculate_ssim(original, stego):

    score, _ = ssim(
        original,
        stego,
        channel_axis=2,
        full=True
    )

    return score


# In[ ]:


mse_val = mse(original, stego)
psnr_val = psnr(original, stego)
ssim_val = calculate_ssim(original, stego)

print("MSE :", mse_val)
print("PSNR:", psnr_val, "dB")
print("SSIM:", ssim_val)


# In[ ]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

orig = np.array(Image.open("original.png"))
stego = np.array(Image.open("LSB_stego.png"))

orig_flat = orig.flatten()
stego_flat = stego.flatten()

plt.figure(figsize=(12,5))

plt.hist(orig_flat,
         bins=256,
         alpha=0.5,
         label="Original")

plt.hist(stego_flat,
         bins=256,
         alpha=0.5,
         label="Stego")

plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Histogram Comparison")
plt.legend()

plt.show()


# In[ ]:


orig_hist,_ = np.histogram(
    orig_flat,
    bins=256,
    range=(0,256)
)

stego_hist,_ = np.histogram(
    stego_flat,
    bins=256,
    range=(0,256)
)

plt.figure(figsize=(12,5))

plt.plot(
    stego_hist - orig_hist
)

plt.title("Histogram Difference")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency Difference")

plt.show()

