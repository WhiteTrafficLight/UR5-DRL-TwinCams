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

import aria.sdk as aria

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import tf
from geometry_msgs.msg import TransformStamped

from common import ctrl_c_handler, quit_keypress, update_iptables

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration
)
from projectaria_tools.core.sensor_data import ImageDataRecord

"""
def yaml_to_CameraInfo(yaml_fname):
    
    Parse a yaml file containing camera calibration data (as produced by 
    rosrun camera_calibration cameracalibrator.py) into a 
    sensor_msgs/CameraInfo msg.
    
    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    
    # Load data from file
    with open(yaml_fname, "r") as file_handle:
        calib_data = yaml.load(file_handle, Loader=yaml.FullLoader)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.K = calib_data["camera_matrix"]["data"]
    camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg
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


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
        
    # Initialize ROS node
    #rospy.init_node('aria_image_publisher', anonymous=True)
    #slam1_pub = rospy.Publisher('/aria/left/image_raw', Image, queue_size=10)
    #slam2_pub = rospy.Publisher('/aria/right/image_raw', Image, queue_size=10)
    #slam1_info_pub = rospy.Publisher('/aria/left/camera_info', CameraInfo, queue_size=10)
    #slam2_info_pub = rospy.Publisher('/aria/right/camera_info', CameraInfo, queue_size=10)
    #camera_info1 = yaml_to_CameraInfo('left_camera.yaml')
    #camera_info2 = yaml_to_CameraInfo('right_camera.yaml')
    #disparity_pub = rospy.Publisher('/aria/disparity',Image,queue_size=10)
    #bridge = CvBridge()    

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
    #rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
    slam1_calib = sensors_calib.get_camera_calib("camera-slam-left")
    slam2_calib = sensors_calib.get_camera_calib("camera-slam-right")
    
    #dst_calib_rgb = get_linear_camera_calibration(512, 512, 150) #,"camera-rgb",rgb_calib.get_transform_device_camera())
    dst_calib1 = get_linear_camera_calibration(512, 512, 150) #,"camera-slam-left",slam1_calib.get_transform_device_camera())
    dst_calib2 = get_linear_camera_calibration(512, 512, 150) #,"camera-slam-right",slam2_calib.get_transform_device_camera())

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
    slam2_window = "SLAM2"
    disparity_window = "Disparity Map"

    cv2.namedWindow(slam1_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slam1_window, 480, 640)
    cv2.setWindowProperty(slam1_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(slam1_window, 50, 50)

    cv2.namedWindow(slam2_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slam2_window, 480, 640)
    cv2.setWindowProperty(slam2_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(slam2_window, 600, 50)
    
    cv2.namedWindow(disparity_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(disparity_window, 480 , 640)
    cv2.setWindowProperty(disparity_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(disparity_window, 50, 1100)
    
    # Initialize stereo block matcher
    
    nDisFactor = 6
    stereo = cv2.StereoBM_create(numDisparities=16*nDisFactor, blockSize=11)
    
    """
    # Initialize stereo block matcher
    window_size = 7
    min_disp = 0
    nDispFactor = 14
    num_disp = 16 * nDispFactor - min_disp

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
    """
   
    
    #rate = rospy.Rate(10)  # 10 Hz
    
    # Set up TF broadcaster
    #tf_broadcaster = tf.TransformBroadcaster()
    
    #slam1_image = observer.images[aria.CameraId.Slam1]
    #output_path = 'slam_left.png' 
    #rgb_image = cv2.cvtColor(observer.images[aria.CameraId.Rgb], cv2.COLOR_RGB2GRAY)
    #output_path2 = 'rgb.png'
    #undistorted_slam1_image = distort_by_calibration(
    #    slam1_image, dst_calib1, slam1_calib
    #)
    #undistorted_slam1_image = np.rot90(undistorted_slam1_image,-1)          
    #undistorted_rgb_image = distort_by_calibration(
    #    rgb_image, dst_calib_rgb, rgb_calib
    #)
    #undistorted_rgb_image = np.rot90(undistorted_rgb_image,-1)
    #cv2.imwrite(output_path2, undistorted_rgb_image)
    #cv2.imwrite(output_path, undistorted_slam1_image)
    
    


    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if (
                aria.CameraId.Slam1 in observer.images
                and aria.CameraId.Slam2 in observer.images
                #and aria.CameraId.Rgb in observer.images
            ):
                slam1_image = observer.images[aria.CameraId.Slam1]
                slam2_image = observer.images[aria.CameraId.Slam2]
                #rgb_image = cv2.cvtColor(observer.images[aria.CameraId.Rgb], cv2.COLOR_RGB2GRAY)
                

                #Apply the undistortion correction
                undistorted_slam1_image = distort_by_calibration(
                    slam1_image, dst_calib1, slam1_calib
                )
                undistorted_slam1_image = np.rot90(cv2.medianBlur(undistorted_slam1_image,5),-1)
                
                undistorted_slam2_image = distort_by_calibration(
                    slam2_image, dst_calib2, slam2_calib
                )
                undistorted_slam2_image = np.rot90(cv2.medianBlur(undistorted_slam2_image,5),-1)
                
                #undistorted_rgb_image = distort_by_calibration(
                #    rgb_image, dst_calib_rgb, rgb_calib
                #)
                #undistorted_rgb_image = np.rot90(cv2.medianBlur(undistorted_rgb_image,5),-1)
                
                
                #undistorted_slam1_image = cv2.cvtColor(undistorted_slam1_image, cv2.COLOR_BGR2GRAY)
                #undistorted_slam2_image = cv2.cvtColor(undistorted_slam2_image, cv2.COLOR_BGR2GRAY)
                
                # Compute the disparity map
                disparity = stereo.compute(undistorted_slam1_image, undistorted_slam2_image) #.astype(np.float32) #/ 16.0
                #disparity = stereo.compute(undistorted_slam1_image, undistorted_rgb_image)
                #disparity = stereo.compute(slam1_image, rgb_image)
                disparity = cv2.medianBlur(disparity, 5)
                
                #Show the undistorted image
                cv2.imshow(slam1_window, undistorted_slam1_image)
                cv2.imshow(slam2_window, undistorted_slam2_image)
                cv2.imshow(disparity_window, np.rot90((disparity - disparity.min()) / (disparity.max() - disparity.min()), -1))#(disparity - disparity.min()) / (disparity.max() - disparity.min()))
                
                """
                # Publish SLAM images to ROS
                try:
                    if len(slam1_image.shape) == 2:
                        slam1_msg = bridge.cv2_to_imgmsg(np.rot90(slam1_image,-1), encoding="mono8")
                    else:
                        slam1_msg = bridge.cv2_to_imgmsg(np.rot90(slam1_image,-1), encoding="bgr8")

                    #if len(slam2_image.shape) == 2:
                    #    slam2_msg = bridge.cv2_to_imgmsg(slam2_image, encoding="mono8")
                    #else:
                    #    slam2_msg = bridge.cv2_to_imgmsg(slam2_image, encoding="bgr8")

                    if len(rgb_image.shape) == 2:
                        slam2_msg = bridge.cv2_to_imgmsg(np.rot90(rgb_image,-1), encoding="mono8")
                    else:
                        slam2_msg = bridge.cv2_to_imgmsg(np.rot90(rgb_image,-1), encoding="bgr8")

                    slam1_pub.publish(slam1_msg)
                    slam2_pub.publish(slam2_msg)
                    slam1_info_pub.publish(camera_info1)
                    slam2_info_pub.publish(camera_info2)
                except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))
                    
                # Publish static transform
                translation = (0.0, 0.0, 0.0)
                rotation = (0.0, 0.0, 0.0, 1.0)
                tf_broadcaster.sendTransform(
                    translation,
                    rotation,
                    rospy.Time.now(),
                    "camera_frame",
                    "map"
                )

                #del observer.images[aria.CameraId.Slam1]
                #del observer.images[aria.CameraId.Rgb]

                rate.sleep()
                """
                
                observer.slam1_image = None
                observer.slam2_image = None

    # 10. Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)


if __name__ == "__main__":
    main()
