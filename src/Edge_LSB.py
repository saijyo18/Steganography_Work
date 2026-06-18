#!/usr/bin/env python
# coding: utf-8

# ## To implement edge based LSB in python from scratch

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# In[3]:


def text_to_binary(text):

    return ''.join(
        format(ord(char), '08b')
        for char in text
    )

def detect_edges_sobel(image_path,
                       threshold=80):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        sobel_x**2 +
        sobel_y**2
    )

    gradient = cv2.normalize(
        gradient,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    gradient = gradient.astype(np.uint8)

    edge_mask = (
        gradient > threshold
    ).astype(np.uint8)

    return edge_mask, gradient


# In[4]:


def encode_edge_lsb(
        image_path,
        secret_message,
        output_path,
        threshold=80):

    img = Image.open(image_path)

    img_array = np.array(img)

    edge_mask, gradient = detect_edges_sobel(
        image_path,
        threshold
    )

    edge_positions = np.argwhere(
        edge_mask == 1
    )

    binary_message = (
        text_to_binary(secret_message)
        + "1111111111111110"
    )

    capacity = len(edge_positions)

    print(
        "Available Edge Pixels:",
        capacity
    )

    print(
        "Bits Needed:",
        len(binary_message)
    )


    if len(binary_message) > capacity:
        raise ValueError(
            "Message too large for edge capacity"
        )

    for i, bit in enumerate(binary_message):

        r, c = edge_positions[i]

        img_array[r, c, 0] = (
            img_array[r, c, 0] & 0xFE
        ) | int(bit)

    stego_img = Image.fromarray(
        img_array.astype(np.uint8)
    )

    stego_img.save(output_path)

    return img_array, edge_mask, gradient



# In[12]:


def decode_edge_lsb_reliable(stego_image_path, edge_mask):
    edge_positions = np.argwhere(edge_mask == 1)


    stego_img = Image.open(stego_image_path)
    stego_array = np.array(stego_img)

    binary_message = ""

    for r, c in edge_positions:
        bit = stego_array[r, c, 0] & 1
        binary_message += str(bit)

        if binary_message.endswith("1111111111111110"):
            break

    if binary_message.endswith("1111111111111110"):
        binary_message = binary_message[:-16]
    else:
        print("Warning: Delimiter not found. The message might be truncated.")

    secret_text = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if len(byte) == 8:
            secret_text += chr(int(byte, 2))

    return secret_text


# In[13]:


## Embed the secret message 

secret="Maybe its the way you say my name Maybe its the way you play your game But its goodI ve never known anybody like youBut its so good, Ive never dreamed of nobody like youAnd Ive heard of a love that comes once in a lifetimeAnd Im pretty sure "
original_img = np.array(
    Image.open("original.png")
)

stego_img, edge_mask, gradient = encode_edge_lsb(
    "original.png",
    secret,
    "edge_stego.png",
    threshold=80
)

print("Embedding Completed")


# In[14]:


extracted_message = decode_edge_lsb_reliable("edge_stego.png", edge_mask)

print("Extracted Message:", extracted_message)


# In[8]:


## Parameters to check the performance of the edge steganography 
def mse(original,
        stego):

    return np.mean(
        (
            original.astype(float)
            -
            stego.astype(float)
        )**2
    )

def psnr(original,
         stego):

    mse_val = mse(
        original,
        stego
    )

    if mse_val == 0:
        return float("inf")

    return 10 * np.log10(
        (255**2) /
        mse_val
    )
def calculate_ssim(
        original,
        stego):

    score, _ = ssim(
        original,
        stego,
        channel_axis=2,
        full=True
    )

    return score


# In[9]:


mse_val = mse(
    original_img,
    stego_img
)

psnr_val = psnr(
    original_img,
    stego_img
)

ssim_val = calculate_ssim(
    original_img,
    stego_img
)

print("MSE :", mse_val)
print("PSNR:", psnr_val, "dB")
print("SSIM:", ssim_val)


# In[ ]:


stego = Image.open("edge_stego.png")

plt.figure(figsize=(8,8))
plt.imshow(stego)
plt.title("Stego Image")
plt.axis("off")
plt.show()


# In[ ]:


stego = Image.open("original.png")

plt.figure(figsize=(8,8))
plt.imshow(stego)
plt.title("Stego Image")
plt.axis("off")
plt.show()


# In[ ]:


## Plot the histogram

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


# print("Embedding Edge Count:",
#       len(edge_positions))
# print("Extraction Edge Count:",
#       len(edge_positions))

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

