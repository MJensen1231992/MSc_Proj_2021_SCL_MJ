import numpy as np
import matplotlib as plt
import cv2


class world:
    def __init__(self, background: str):

        img = cv2.imread(background, cv2.IMREAD_GRAYSCALE)
        self.background = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)[1]
        
        cv2.imshow('map', img)
        



def main():
    print(cv2.__version__)
    show = world('sideProjects/GIS_Extraction/plots/GIS_map3.png')
    # show = world('robsim_parent/workspaces/office_workspace_1.png')
    while True:
            k = cv2.waitKey(0) & 0xFF
            print(k)
            if k == 27:
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()

