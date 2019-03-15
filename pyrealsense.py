import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale(0.001)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.5 #meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame :
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        grey_color = 3000

        depth_image2 = np.where((depth_image > clipping_distance) | (depth_image <= 0), grey_color, depth_image)
        sorted_depth_vector = np.argpartition(depth_image2,25,None)
        sorted_depth_vector = sorted_depth_vector[:20]
        min_depth_col = sorted_depth_vector/640
        min_depth_row = sorted_depth_vector%640
        col_centroid = (np.sum(min_depth_col))/20
        row_centroid = (np.sum(min_depth_row))/20
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        bg_removed1 = bg_removed
        cv2.circle(bg_removed1,(row_centroid,col_centroid), 25, (0,0,255), -1)
        cv2.imshow('rgb_image' , bg_removed1)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('c') or key == 27:
            cv2.destroyAllWindows()
            break

    print("calibrated values",row_centroid,col_centroid)
    roi_radius = 10

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame :
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        grey_color = 3000
        depth_image2 = np.where((depth_image > clipping_distance) | (depth_image <= 0), grey_color, depth_image)
        print("depth image 2 shape",depth_image2.shape)
        row1 = row_centroid - roi_radius
        row2 = row_centroid + roi_radius
        col1 = col_centroid - roi_radius
        col2 = col_centroid + roi_radius
        print('row1,row2,col1,col2',row1,row2,col1,col2)
        print('col centroid', col_centroid)
        roi_depth_array = depth_image2[row1:row2,col1:col2]
        print('its shape ',roi_depth_array.shape)
        print('roi_depth_array',roi_depth_array)
        sorted_roi_depth_vector = np.argpartition(roi_depth_array,25,None)
        sorted_roi_depth_vector = sorted_roi_depth_vector[:20]
        print('sorted_roi_depth_vector',sorted_roi_depth_vector)
        min_roi_depth_col = sorted_roi_depth_vector/(2*roi_radius)
        min_roi_depth_row = sorted_roi_depth_vector%(2*roi_radius)
        print('minimum row',min_roi_depth_row)
        print('minimum col',min_roi_depth_col)

        new_centroid_col = (np.sum(min_roi_depth_col))/20
        new_centroid_row = (np.sum(min_roi_depth_row))/20
        print("after sum :",new_centroid_row,new_centroid_col)
        row_centroid = new_centroid_row + row_centroid - roi_radius
        col_centroid = new_centroid_col + col_centroid - roi_radius
        print(row_centroid,col_centroid)

        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        bg_removed1 = bg_removed
        cv2.circle(bg_removed1,(col_centroid,row_centroid), 25, (0,0,255), -1)
        cv2.imshow('Finally following' , bg_removed1)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


finally:
    pipeline.stop()
