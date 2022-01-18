import cv2 
import matplotlib.pyplot as plt 
import numpy as np
import pydicom

#show the image
def plot_big(img, cmap="Greys_r"):
  plt.figure(figsize=(10, 10))
  plt.imshow(img, cmap=cmap)
  

#mascara para retirar os ossos 
def biggest_component(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    mask = np.zeros(output.shape)
    mask[output == max_label] = 255
    return mask


#aplicação da alisamento  
def apply_smoothing(values, window_size=15):
  smoothed = []
  for i in range(1, len(values)):
    l = i - window_size // 2
    r = i + window_size // 2
    window = values[l : r]
    smoothed.append(np.mean(window))
  return smoothed


def find_extrema_with_windowing(values, window_size=21, is_min=False): 
  arg_fn = np.argmin if is_min else np.argmax
  comp_fn = less_than_fn if is_min else greater_than_fn

  extrema_points = []
  #for idx in range(window_size//2, len(values)-window_size//2):
  for idx in range(len(values)):
    l = idx - window_size // 2
    r = idx + window_size // 2
    window = values[l : r]
    if len(window) <= 1:
      continue
    local_extrema_idx = arg_fn(window)
    if idx == (l + local_extrema_idx):
      avg_before = np.mean(window[:len(window)//2])
      avg_after = np.mean(window[len(window)//2:])
      if comp_fn(values[idx], avg_before) and comp_fn(values[idx], avg_after):
        extrema_points.append(idx)

  return extrema_points


#picos no cerebro 
def find_normal_brain_peaks(smoothed_histogram, maxs):
  normal_maxs = []
  for m in maxs:
    if m > 1500:
      normal_maxs.append(m)
  return normal_maxs


def check_for_hemorrhages(smoothed_histogram, maxs, normal_maxs):
  possible_hemorrhages
  for m in maxs:
    if (m > normal_maxs[-1]):
      pass # para terminar


# assume that small connected components are noise and discard them
def mask_out_small_components(labels, stats, area_threshold=20):
  components_to_keep = []
  for label_idx in range(1, num_labels): # starts from 1 to ignore backgronud
    if stats[label_idx, cv2.CC_STAT_AREA] >= area_threshold:
      components_to_keep.append(label_idx)
    
  mask = np.zeros_like(labels)
  for label_idx in components_to_keep:
    mask[labels == label_idx] = 255
  
  return mask




#definição da image que vai ser usada, semdo passada para GRAYSCALE
img = cv2.imread("E:\\hd1\\rsna-hemorrhage\\genesis-brain-hemorrhage-main\\aaaq.png",cv2.IMREAD_GRAYSCALE)


#definição do hist da imagem 
histogram = cv2.calcHist([img], [0], None, [256], [0, 256])


# definição do intervalo do threshold
mask = np.zeros_like(img) + 255
a,thresh3 = cv2.threshold(img,30,210,cv2.THRESH_BINARY)
cv2.imshow("janela",thresh3)
cv2.waitKey(0)
'''print(mask)
mask[np.zeros_like(img) > 210] = 0
mask[np.zeros_like(img) < 30] = 0
'''
#definição do hist da imagem (com o threshold)
histogram = cv2.calcHist([img], [0], mask, [256], [0, 256])


#erosão e dilatação 
masked_img = np.copy(img)
masked_img[mask == 0] = 0
masked_img = cv2.erode(masked_img, np.ones((3, 3)))
masked_img = cv2.dilate(masked_img, np.ones((3, 3)))


#mascara
brain_mask = biggest_component(masked_img)
#brain_mask = cv2.erode(brain_mask, np.ones((5, 5)))
#brain_mask = cv2.dilate(brain_mask, np.ones((5, 5)))
masked_brain = np.copy(masked_img) 
masked_brain[brain_mask == 0] = 0

#definição do hist da imagem (com mascara)
histogram = cv2.calcHist([img], [0], brain_mask.astype(np.uint8), [256], [0, 256])


less_than_fn = lambda x, y: x < y
greater_than_fn = lambda x, y: x > y


smoothed_histogram = apply_smoothing(histogram, window_size=9)
maxs = find_extrema_with_windowing(smoothed_histogram, window_size=7, is_min=False)
mins = find_extrema_with_windowing(smoothed_histogram, window_size=7, is_min=True)

plt.figure(figsize=(15, 7))
plt.plot(histogram)
plt.plot(smoothed_histogram)
plt.legend(['raw', 'smoothed'])
plt.plot(maxs, [0] * len(maxs), '.r', markersize=10)
#plt.plot(mins, [0] * len(mins), '.g', markersize=10)

#mostrar o histograma com os pontos de máximo
plt.plot(histogram[150:])
plt.plot(smoothed_histogram[150:])
high_maxs = [m-150 for m in maxs if m > 150]
plt.plot(high_maxs, [0] * len(high_maxs), '.r', markersize=10)


# find the first maximum after the 150 and assume it is the hemorrhage
hemorrhage_idx = 0
for i, val in enumerate(maxs):
  if val > 150:
    hemorrhage_idx = i
    break

threshold = int(0.5 * maxs[hemorrhage_idx] + 0.5 * maxs[hemorrhage_idx-1])

width = maxs[-1] - threshold
left_limit = maxs[-1] - width
right_limit = maxs[-1] + width
print(left_limit, maxs[-1], right_limit)

mask = np.copy(masked_brain)
mask[mask < left_limit] = 0
mask[mask > right_limit] = 0
mask[mask > 0] = 255

mask = cv2.erode(mask, np.ones((5, 5)))
#mask = cv2.dilate(mask, np.ones((3, 3)), iterations=2)
segmented = np.copy(img)
segmented[mask == 0] = 0

# draw connected components and their areas
output = cv2.connectedComponentsWithStats(mask)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]
components_view = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
for label_idx in range(1, num_labels): # starts from 1 to ignore backgronud
  bb_left = stats[label_idx, cv2.CC_STAT_LEFT]
  bb_top = stats[label_idx, cv2.CC_STAT_TOP]
  bb_height = stats[label_idx, cv2.CC_STAT_HEIGHT]
  bb_width = stats[label_idx, cv2.CC_STAT_WIDTH]
  bb_area = stats[label_idx, cv2.CC_STAT_AREA]

  cv2.rectangle(components_view, (bb_left, bb_top), (bb_left + bb_width, bb_top + bb_height), (255, 0, 0), 1)
  cv2.putText(components_view, f"{bb_area}", (bb_left, bb_top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))


noise_removal_mask = mask_out_small_components(labels, stats)
mask[noise_removal_mask == 0] = 0
segmented[noise_removal_mask == 0] = 0

plt.figure(figsize=(10, 40))
plt.subplot(4, 1, 1)
plt.imshow(img, cmap="Greys_r")
plt.subplot(4, 1, 2)
plt.imshow(masked_brain, cmap="Greys_r")
plt.subplot(4, 1, 3)
plt.imshow(components_view, cmap="Greys_r")
plt.subplot(4, 1, 4)
plt.imshow(segmented, cmap="Greys_r")


heatmap = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
heatmap[mask > 0] = (255, 0, 0)

print(mask.shape, np.min(mask), np.max(mask))
print(heatmap.shape, np.min(heatmap), np.max(heatmap))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="Greys_r")
plt.subplot(1, 2, 2)
plt.imshow(heatmap, cmap="Greys_r")