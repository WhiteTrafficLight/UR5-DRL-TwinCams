# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
from collections import deque


import aria.sdk as aria

import cv2
import numpy as np
from common import ctrl_c_handler, quit_keypress, update_iptables

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration
)
from projectaria_tools.core.sensor_data import (
    ImageDataRecord,
    MotionData
)    

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


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables() 

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
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
    # Note: by default streaming uses Wifi
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # 5. Get sensors calibration
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
    
    dst_calib_rgb = get_linear_camera_calibration(512, 512, 150) #,"camera-rgb",rgb_calib.get_transform_device_camera())

    # 6. Start streaming
    streaming_manager.start_streaming()

    # 7. Configure subscription to listen to Aria's RGB stream.
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.Imu
    )
    streaming_client.subscription_config = config
     
    class StreamingClientObserver:
        def __init__(self):
            self.image_queue = deque(maxlen=10)  # Store up to 10 images

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.image_queue.append(image)

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 9. Render the streaming data until we close the window
    rgb1_window = "RGB1"
    rgb2_window = "RGB2"
    disparity_window = "Disparity Map"

    cv2.namedWindow(rgb1_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb1_window, 512, 512)
    cv2.setWindowProperty(rgb1_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb1_window, 50, 50)

    cv2.namedWindow(rgb2_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb2_window, 512, 512)
    cv2.setWindowProperty(rgb2_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb2_window, 600, 50)
    
    cv2.namedWindow(disparity_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(disparity_window, 512, 512)
    cv2.setWindowProperty(disparity_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(disparity_window, 50, 700)
    
    # Initialize stereo block matcher
    #nDisFactor = 6
    #stereo = cv2.StereoBM_create(numDisparities=16*nDisFactor, blockSize=11)
    
    
    # Initialize stereo block matcher
    window_size = 7
    min_disp = 0
    nDispFactor = 14
    num_disp = 16 #* nDispFactor - min_disp

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
    
    
    # Create WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    while True:
        if len(observer.image_queue) >= 2:
            rgb1_image = cv2.cvtColor(observer.image_queue[-1], cv2.COLOR_RGB2GRAY)  # Latest image
            rgb2_image = cv2.cvtColor(observer.image_queue[0], cv2.COLOR_RGB2GRAY)   # Oldest image in the queue

            # Apply the undistortion correction        
            undistorted_rgb1_image = distort_by_calibration(rgb1_image, dst_calib_rgb, rgb_calib)
            undistorted_rgb1_image = np.rot90(undistorted_rgb1_image, -1)

            undistorted_rgb2_image = distort_by_calibration(rgb2_image, dst_calib_rgb, rgb_calib)
            undistorted_rgb2_image = np.rot90(undistorted_rgb2_image, -1)

            # Compute the disparity map
            disparity = stereo.compute(undistorted_rgb1_image, undistorted_rgb2_image)
            disparity = np.int16(disparity)  # Convert disparity to 16-bit signed integers for WLS filter

            # Compute the disparity maps
            left_disparity = stereo.compute(undistorted_rgb1_image, undistorted_rgb2_image)
            right_disparity = right_matcher.compute(undistorted_rgb2_image, undistorted_rgb1_image)
            left_disparity = np.int16(left_disparity)  # Convert disparity to 16-bit signed integers for WLS filter
            right_disparity = np.int16(right_disparity)

            # Apply WLS filter
            filtered_disparity = wls_filter.filter(left_disparity, undistorted_rgb1_image, disparity_map_right=right_disparity)
            filtered_disparity = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            filtered_disparity = np.uint8(filtered_disparity)

            # Show the images
            cv2.imshow(rgb1_window, undistorted_rgb1_image)
            cv2.imshow(rgb2_window, undistorted_rgb2_image)
            cv2.imshow(disparity_window, filtered_disparity)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



