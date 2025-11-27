import numpy as np
import os
# confidence numpy를 하나의 파일이 아닌 전체 폴더에서 모두 불러와서 계산하기 
confidence_folder = "/home/shinds/my_document/DLFromScratch5/test/vae/sss/mtl_segformer/depth-anything-3/depth_output_zed/not_move/confidence"
confidence_files = os.listdir(confidence_folder)
confidence_files.sort()
confidence_npys = [np.load(os.path.join(confidence_folder, file)) for file in confidence_files]
confidence = np.concatenate(confidence_npys)
# 각 픽셀의 confidence 값에 따른 비율 구하기(1~100 설정)
for i in range(0, 101):
    print(f"{i}: {(confidence >= i).sum() / confidence.size}")