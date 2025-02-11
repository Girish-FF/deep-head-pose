import os
import numpy as np

path_dir = r"AFLW2000"
# path_dir = r"C:\\Users\\Girish\\Desktop\\FF-Projects\\Datasets\\testFaceNonface\\Face"

# text_file_path = os.path.join("C:\\Users\\Girish\\Desktop\\FF-Projects\\MatchScoreAnalysis\\NexDataCropped\\femalecaucassian01719.txt")
text_file_path = "AFLW2000.txt"
# with open(text_file_path, 'w') as file:
#     for folder in os.listdir(path_dir):
#         # if folder !="InDoubt": continue
#         if os.path.isdir(os.path.join(path_dir, folder)): 
#             # for subfolder in os.listdir(os.path.join(path_dir, folder)):
#                 for filename in os.listdir(os.path.join(path_dir, folder)):
#                     if filename.endswith((".jpg", ".png", ".jpeg", ".tiff", ".JPG", ".PNG", ".JPEG")):
#                         # print(os.path.join(folder,filename))
#                         # break
#                         # file.write(f"{filename.split('.')[0]}\n")
#                         # file.write(os.path.join(folder,f"{os.path.splitext(filename)[0]}\n"))
#                         # file.write(os.path.join(f"{filename}\n"))
#                         file.write(os.path.join(folder,f"{filename}\n"))
# file.close()

with open(text_file_path, "w") as file:
     for filename in os.listdir(path_dir):
          if filename.endswith((".jpg", ".png", ".jpeg", ".tiff", ".JPG", ".PNG", ".JPEG")):
               file.write(os.path.join(f"{filename.split('.')[0]}\n"))

file.close()

################################################################################################
# # path_dir = "C:\\Users\\Girish\\Desktop\\FF-Projects\\deep-head-pose\\300W_LP"
# path_dir = "C:\\Users\\Girish\\Desktop\\FF-Projects\\Datasets\\testFaceNonface"
# text_file_path = os.path.join('C:\\Users\\Girish\\Desktop\\FF-Projects\\Datasets\\files2.csv')
# data = []

# for folder in os.listdir(path_dir):
#     for filename in os.listdir(os.path.join(path_dir, folder)):
#         if filename.endswith(("jpg", "png", "jpeg", "tiff")):
#             print(os.path.join(path_dir, folder, filename))
#             # print(os.path.join(folder,f"{filename.split('.')[0]}\n"))
#             if folder=="Nonface":
#                 fold=1
#             else:
#                 fold=0
#             data.append([os.path.join(path_dir,folder, filename), fold])

# # with open(text_file_path, 'w') as file:
# #     # file.write(os.path.join(folder,f"{filename.split('.')[0]}\n"))
# #     file.write(str(data))
# # file.close()
# import pandas as pd
# # data = pd.DataFrame(data)
# # print((data.shape))

# # data.to_csv(text_file_path, index=False)   
# np.savetxt(text_file_path, data, fmt='%s')     




