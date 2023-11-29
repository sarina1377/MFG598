import cv2
import numpy as np
import csv
import os

# Declare global variables
poly_points = []
ix=0
iy=0

### In this script we extract pixel values from polygons on the image. The vegeation is then defined by the user, and features values 
### are written to csv train file

#### Function to draw polygons on the image: use left buttom to pick three points, then click the right buttom to draw the polygon

def draw_polygon(event, x, y, flags, param):
    ### gloabal variables are applied as this function is called everytime the mouse is clicked
    global ix, iy, drawing, poly_points, img_draw

    ### check is the left button is pressed: if yes, pick the point and add to the list of points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        poly_points.append((x, y))

    ### check is the left button is not pressed: it will draw a circle at the point of mouse to show the selected point

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img_draw, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', img_draw)

    ### checks if the right button is clicked. if yes, the selction of the points is over and it will draw lines between points to
    ### show the final polygon

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(poly_points) > 1:
            cv2.polylines(img_draw, [np.array(poly_points)], isClosed=True, color=(0, 255, 0), thickness=2)
            ### pixel extraction from the polygons:
            ### here we create a mask with same dimensions as the img
            mask = np.zeros_like(img[:,:,0])
            ### fill in the polygon with white. now mask will be white only in the selected polygon
            cv2.fillPoly(mask, [np.array(poly_points)], 255)
            ### get coords where the mask is white== polygon
            y_coords, x_coords = np.where(mask == 255)
            polygon_pixels = img[y_coords, x_coords]

            for (y, x), pixel in zip(zip(y_coords, x_coords), polygon_pixels):
                r, g, b = pixel
                print(f"Coordinates=(Y={y}, X={x}), RGB=(R={r}, G={g}, B={b})")

            ### the user will now define the vegetation: 0 for non, 1 for vegetation

            classification=input("Insert vegetation:")
            #classification = "non-vegetation" if np.mean(polygon_pixels[:, 1]) > 100 else "vegetation"
            print(f"Classification: {classification}")
            ### Saving the R,G,B,EXG,EXGR,CIVE to the train file
            file_path = r"C:\Users\sarin\Dropbox (ASU)\CEE\MFG 598\project\train_video.csv"
            ### check if the file already exists:
            write_header = not os.path.exists(file_path)
            with open(file_path, 'a', newline='') as datafile:
                content = csv.writer(datafile)
                ### the tricky part!: write the header only if the file already doesn't exist
                if write_header:
                    content.writerow(['R', 'G', 'B', 'ExG', 'ExGR', 'CIVE', 'Classification'])

                for (y, x), pixel in zip(zip(y_coords, x_coords), polygon_pixels):
                    r, g, b = pixel
                    exg = 2 * g - r - b
                    exgr = exg - (1.4 * r - g)
                    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
                    content.writerow([r, g, b, exg, exgr, cive, classification])

            poly_points.clear()
            cv2.imshow('image', img_draw)


#######path: C:\Users\sarin\Dropbox (ASU)\PC (2)\Downloads\satelliteview.jpg
# Load image and create a copy for drawing
## note that the user can use multiple images to get the train data simply by running the code again and modifying the path 
image_path = input("Enter the path to the image: ")
img = cv2.imread(image_path)
img_draw = img.copy()

# Check if the image is loaded properly
if img is None:
    print("Error: Image not found. Please check the path.")

    #### image is imported propery, we draw the polygons:
else:
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_polygon)

    while True:
        cv2.imshow('image', img_draw)
        k = cv2.waitKey(1) & 0xFF
        #### Esc key to stop
        if k == 27:  
            break

    cv2.destroyAllWindows()
