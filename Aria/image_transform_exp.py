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
from projectaria_tools.core import data_provider, calibration

from common import ctrl_c_handler, quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration
)
from projectaria_tools.core.sensor_data import ImageDataRecord

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

def undistort_and_rectify(image, K, D):
    h, w = image.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)
    rectified_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    return rectified_image

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
    
    D_rgb = np.array(rgb_calib.projection_params)
    print(D_rgb)
    # Use the same intrinsic parameters for SLAM1
    K_slam = K_rgb
    D_slam = D_rgb

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

    cv2.namedWindow(slam1_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slam1_window, 512, 512)
    cv2.setWindowProperty(slam1_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(slam1_window, 50, 50)

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 512, 512)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 600, 50)
    
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

                # Rectify images using the same intrinsic parameters
                rect_slam1_image = undistort_and_rectify(undistorted_slam1_image, K_slam, D_slam)
                rect_rgb_image = undistort_and_rectify(undistorted_rgb_image, K_rgb, D_rgb)

                # Show the images
                cv2.imshow(slam1_window, rect_slam1_image)
                cv2.imshow(rgb_window, rect_rgb_image)

                observer.slam1_image = None
                observer.rgb_image = None

    # Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)

if __name__ == "__main__":
    main()




