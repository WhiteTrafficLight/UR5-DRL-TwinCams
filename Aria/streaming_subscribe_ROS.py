import argparse
import sys

import aria.sdk as aria

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from common import quit_keypress, update_iptables

from projectaria_tools.core.sensor_data import ImageDataRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # Initialize ROS node
    rospy.init_node('aria_image_publisher', anonymous=True)
    slam1_pub = rospy.Publisher('/aria/left/slam', Image, queue_size=10)
    slam2_pub = rospy.Publisher('/aria/right/slam', Image, queue_size=10)
    bridge = CvBridge()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create StreamingClient instance
    streaming_client = aria.StreamingClient()

    #  2. Configure subscription to listen to Aria's RGB and SLAM streams.
    # @see StreamingDataType for the other data types
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.Slam
    )

    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.Slam] = 1

    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 3. Create and attach observer
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    # 4. Start listening
    print("Start listening to image data")
    streaming_client.subscribe()

    # 5. Visualize the streaming data until we close the window
    rgb_window = "Aria RGB"
    slam_window = "Aria SLAM"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(slam_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slam_window, 480 * 2, 640)
    cv2.setWindowProperty(slam_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(slam_window, 1100, 50)

    rate = rospy.Rate(10)  # 10 Hz

    while not quit_keypress() and not rospy.is_shutdown():
        # Render the RGB image
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imshow(rgb_window, rgb_image)
            del observer.images[aria.CameraId.Rgb]

        # Stack and display the SLAM images, and publish them to ROS
        if (
            aria.CameraId.Slam1 in observer.images
            and aria.CameraId.Slam2 in observer.images
        ):
            slam1_image = np.rot90(observer.images[aria.CameraId.Slam1], -1)
            slam2_image = np.rot90(observer.images[aria.CameraId.Slam2], -1)
            cv2.imshow(slam_window, np.hstack((slam1_image, slam2_image)))

            # Publish SLAM images to ROS
            try:
                if len(slam1_image.shape) == 2:
                    slam1_msg = bridge.cv2_to_imgmsg(slam1_image, encoding="mono8")
                else:
                    slam1_msg = bridge.cv2_to_imgmsg(slam1_image, encoding="bgr8")

                if len(slam2_image.shape) == 2:
                    slam2_msg = bridge.cv2_to_imgmsg(slam2_image, encoding="mono8")
                else:
                    slam2_msg = bridge.cv2_to_imgmsg(slam2_image, encoding="bgr8")

                slam1_pub.publish(slam1_msg)
                slam2_pub.publish(slam2_msg)
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))

            del observer.images[aria.CameraId.Slam1]
            del observer.images[aria.CameraId.Slam2]

        rate.sleep()

    # 6. Unsubscribe to clean up resources
    print("Stop listening to image data")
    streaming_client.unsubscribe()


if __name__ == "__main__":
    main()
