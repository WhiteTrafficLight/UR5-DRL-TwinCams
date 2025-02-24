import argparse
import sys
import aria.sdk as aria
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tf
from scipy.spatial.transform import Rotation as R

from common import ctrl_c_handler, quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration
)
from projectaria_tools.core.sensor_data import ImageDataRecord

"""
import torch

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Zoe_K
model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)

# Zoe_NK
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()

def transform_image_to_rgb_perspective(image, K_slam, K_rgb, R_slam_rgb, T_slam_rgb):
    h, w = image.shape[:2]
    
    # Create the transformation matrix
    R_T = np.hstack((R_slam_rgb, T_slam_rgb.reshape(-1, 1)))
    P_slam_to_rgb = R_T #K_rgb @ R_T
    
    # Ensure the matrix is of the correct type and shape
    P_slam_to_rgb = P_slam_to_rgb.astype(np.float32)
    
    # Convert to 3x3 by adding a row [0, 0, 1] if necessary
    if P_slam_to_rgb.shape == (3, 4):
        P_slam_to_rgb = P_slam_to_rgb[:, :3]
    
    transformed_image = cv2.warpPerspective(image, P_slam_to_rgb, (w, h))

    return transformed_image

def rectify_images(undistorted_slam1_image, undistorted_rgb_image, K1, K2):
    h, w = undistorted_rgb_image.shape[:2]

    # Stereo rectify
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=K1, distCoeffs1=None, 
        cameraMatrix2=K2, distCoeffs2=None,
        imageSize=(w, h), R=np.eye(3), T=np.zeros(3),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, None, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, None, R2, P2, (w, h), cv2.CV_32FC1)
    rect_slam1_image = cv2.remap(undistorted_slam1_image, map1x, map1y, cv2.INTER_LINEAR)
    rect_rgb_image = cv2.remap(undistorted_rgb_image, map2x, map2y, cv2.INTER_LINEAR)
    
    return rect_slam1_image, rect_rgb_image

def main():
    
    
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
        
    bridge = CvBridge()    

    # Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect to the device
    device = device_client.connect()

    # 3. Retrieve the device streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # 4. Use a custom configuration for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # 5. Get sensors calibration
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
    slam1_calib = sensors_calib.get_camera_calib("camera-slam-left")
    
    # Extracting calibration parameters
    K_rgb = np.array([[rgb_calib.get_focal_lengths()[0], 0, rgb_calib.get_principal_point()[0]],
                   [0, rgb_calib.get_focal_lengths()[1], rgb_calib.get_principal_point()[1]],
                   [0, 0, 1]])
    K_slam = np.array([[slam1_calib.get_focal_lengths()[0], 0, slam1_calib.get_principal_point()[0]],
                   [0, slam1_calib.get_focal_lengths()[1], slam1_calib.get_principal_point()[1]],
                   [0, 0, 1]])         

    # Relative pose (translation and rotation) between the RGB and SLAM cameras
    T_slam_rgb = np.array([-0.00411119, -0.0120447, -0.00540373])
    quaternion_rgb = [0.331849, 0.0375742, 0.0387809, 0.941786]
    R_slam_rgb = R.from_quat(quaternion_rgb).as_matrix()

    dst_calib_rgb = get_linear_camera_calibration(512, 512, 150)
    dst_calib_slam = get_linear_camera_calibration(512, 512, 150)

    # 6. Start streaming
    streaming_manager.start_streaming()

    # 7. Configure subscription to listen to Aria's RGB stream.
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.Slam
    )
    streaming_client.subscription_config = config
     
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image        

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 9. Render the streaming data until we close the window
    slam1_window = "SLAM1"
    rgb_window = "RGB"
    disparity_window = "Disparity Map"

    cv2.namedWindow(slam1_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slam1_window, 512, 512)
    cv2.setWindowProperty(slam1_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(slam1_window, 50, 50)

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 512, 512)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 600, 50)
    
    cv2.namedWindow(disparity_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(disparity_window, 512, 512)
    cv2.setWindowProperty(disparity_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(disparity_window, 50, 700)

    window_size = 7
    min_disp = 0
    num_disp = 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)
    
    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if (
                aria.CameraId.Slam1 in observer.images
                and aria.CameraId.Rgb in observer.images
            ):
                slam1_image = observer.images[aria.CameraId.Slam1]
                rgb_image = cv2.cvtColor(observer.images[aria.CameraId.Rgb], cv2.COLOR_RGB2GRAY)

                # Apply the undistortion correction
                undistorted_slam1_image = distort_by_calibration(
                    slam1_image, dst_calib_slam, slam1_calib
                )
                undistorted_slam1_image = np.rot90(undistorted_slam1_image, -1)

                undistorted_rgb_image = distort_by_calibration(
                    rgb_image, dst_calib_rgb, rgb_calib
                )
                undistorted_rgb_image = np.rot90(undistorted_rgb_image, -1)

                # Transform SLAM1 image to RGB perspective
                transformed_slam1_image = transform_image_to_rgb_perspective(
                    undistorted_rgb_image, K_slam, K_rgb, R_slam_rgb, T_slam_rgb
                )

                # Rectify images
                #rect_slam1_image, rect_rgb_image = rectify_images(transformed_slam1_image, undistorted_rgb_image, K_slam, K_rgb)

                # Compute disparity map
                #left_disparity = stereo.compute(rect_slam1_image, rect_rgb_image)
                #right_disparity = right_matcher.compute(rect_rgb_image, rect_slam1_image)
                #left_disparity = np.int16(left_disparity)
                #right_disparity = np.int16(right_disparity)

                #filtered_disparity = wls_filter.filter(left_disparity, rect_slam1_image, disparity_map_right=right_disparity)
                #filtered_disparity = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                #filtered_disparity = np.uint8(filtered_disparity)
                
                depth_numpy = zoe.infer_pil(undistorted_rgb_image )  
                
                
                # Show the images
                #cv2.imshow(slam1_window, transformed_slam1_image)
                cv2.imshow(rgb_window, depth_numpy)
                #
                # cv2.imshow(disparity_window, filtered_disparity)

                observer.slam1_image = None
                observer.rgb_image = None

    # 10. Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)

if __name__ == "__main__":
    main()






