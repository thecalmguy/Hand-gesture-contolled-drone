import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

kernel = np.ones((5,5),np.uint8)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame :
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # print(depth_image.shape)
        # print(color_image.shape)
        grey_color = 3000

        depth_image2 = np.where((depth_image > clipping_distance) | (depth_image <= 0), grey_color, depth_image)
        sorted_depth_vector = np.argpartition(depth_image2,25,None)
        sorted_depth_vector2 = sorted_depth_vector[:20]
        min_depth_col = sorted_depth_vector2/640
        min_depth_row = sorted_depth_vector2%640
        col_centroid = (np.sum(min_depth_col))/20
        row_centroid = (np.sum(min_depth_row))/20
        print(row_centroid,col_centroid)
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        # print(depth_image[:10,:10])
        # print("printed once")
        bg_removed1 = bg_removed
        cv2.circle(bg_removed1,(row_centroid,col_centroid), 25, (0,0,255), -1)
        cv2.imshow('rgb_image' , bg_removed1)
        gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, 0)

        thresh = cv2.blur(thresh, (5,5))
        # thresh = cv2.dilate(thresh,kernel,iterations=1)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cv2.imshow('Image', thresh)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
