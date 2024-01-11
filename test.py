import cv2
import matplotlib.pyplot as plt
import numpy as np
# im_gray = cv2.imread('data/input_images/406415526_1664281294101677_6627960005621645181_n.png', 0)
# (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imwrite('bw_image.png', im_bw)
genders = ['Mężczyzna','Kobieta','Kobieta','Mężczyzna','Mężczyzna','Mężczyzna','Mężczyzna','Mężczyzna','Mężczyzna','Kobieta','Mężczyzna','Kobieta','Mężczyzna','Mężczyzna']
image_answers = [[['ffffff','bbff00','0015ff','ffea00','bfef67','7afa0a','b6ff47','40d41c','91ff00','068ddb','eaff00','1495ff','3d81ff','538b37'],
          ['ffffff','0044ff','aeff00','ff4000','0096e0','10652b','13a049','1b74cf','008509','ff4b14','004cff','ff4000','ff4000','98d562']],
[['5a8a2e','ffae00','fa8e00','ffffff','a6651c','663f15','e08700','6a9652','704000','986210','c66a39','77ac53','e0812e','a84300'],
          ['ff6600','00a334','3d0000','c25a00','ffffff','3b8d1b','479900','7a3312','fcc764','35762d','6f9e00','d4852b','140000','8ab97e']],
[['c7c2c2','ff8c00','ec973c','ffa200','608b9f','00496b','5d5756','213446','0c0b28','e7b85a','b1a14e','232448','d1a000','dda978'],
          ['ffa200','8e8d85','3f380d','2e00e6','eab76c','e5ebf0','ffdaa3','d5a076','000f2e','5375ac','5e00ff','cbab62','000b3d','393979']],
[['f8f1f1','872f12','9e0000','ff7700','565348','7c3509','c77400','424343','7a7c85','d78419','ff8800','d4a563','f6bc1e','a54627'],
          ['050e29','492c2c','cb7e0b','05e9fa','9a5a23','0b1d23','380000','b98359','6b4d00','31425e','ff7700','2e2f3d','1f0000','6a6862']],
[['ffffff','cc8500','f8efce','ff7700','3f799d','249bae','8a9493','32424e','6f7890','dc8960','ff6a00','57607a','35afe3','5598b9'],
          ['6cb3cb','60789f','f09e6a','429ebd','ebb19e','cc3300','ffffff','a88c86','fff1eb','398721','ffffff','e47225','fbb450','f5f4f4']]]

images = ['Red_eyed_tree_frog_edit2.jpg', 'steve-payne-v8-W4FQvtB4-unsplash.jpg','full_moon.jpg','marcell-rubies-GAp2GNAr128-unsplash (1).jpg','abdullah-raafat-rWaQGsOquuc-unsplash.jpg']



# rgb_values = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# rgb_strings = ['ff0000', '00ff00', '0000ff']
current_image_index = 4
rgb_values = [tuple(int(rgb[i:i+2], 16) for i in (0, 2, 4)) for rgb in image_answers[current_image_index][0]]
rgb_values1 = [tuple(int(rgb[i:i+2], 16) for i in (0, 2, 4)) for rgb in image_answers[current_image_index][1]]
rgb_values2 = [[[201, 75, 21] , [86, 123, 54] , [23, 129, 219], [47, 56, 62], [67, 79, 76]],
               [[153, 91, 39] , [66 ,96 ,28], [234, 232 ,226], [50 ,43 ,26], [214, 181, 143]],
[[210, 162, 118], [51, 68, 81], [100, 102, 114], [16, 27, 38], [92, 100, 115]],
[[137, 101, 75 ], [39, 48 ,49],[29, 32 ,30], [70, 70, 67], [98, 96, 92]],
[[166, 89 ,91] , [41 ,54 ,64], [196, 187, 192], [31, 31 ,35], [197 ,157 ,143]]
               ]

image_path = 'data/input_images/'+images[current_image_index]

# fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 5))
ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 5), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 5), (2, 0), colspan=2)
ax4 = plt.subplot2grid((3, 5), (0, 2), rowspan=3, colspan=3)

rgb_array = np.array(rgb_values) / 255.0  # Normalize RGB values to [0, 1]
ax1.imshow([rgb_array])
ax1.set_title('Odpowiedz 1')

rgb_array1 = np.array(rgb_values1) / 255.0  # Normalize RGB values to [0, 1]
ax2.imshow([rgb_array1])
ax2.set_title('Odpowiedz 2')

rgb_array2 = np.array(rgb_values2[current_image_index]) / 255.0
ax3.imshow([rgb_array2])
ax3.set_title('Wynik programu')

image = plt.imread(image_path)
ax4.imshow(image)
ax4.set_title('Obraz '+str(current_image_index+1))

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')

plt.show()