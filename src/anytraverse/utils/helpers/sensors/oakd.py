import depthai as dai
import numpy as np
from numpy import typing as npt


type Image = npt.NDArray[np.uint8]
type Pointcloud = npt.NDArray[np.float32]


class OakdCameraManager:
    _pcl_config_input: dai.node.XLinkIn
    _device: dai.Device
    _queue: dai.DataOutputQueue
    _rgb_stream_name: str
    _pointcloud_stream_name: str

    def __init__(
        self,
        pipeline: dai.Pipeline,
        rgb_stream_name: str = "rgb",
        pointcloud_stream_name: str = "points",
        min_depth: float = 0.25,
        max_depth: float = 10.0,
    ) -> None:
        self._rgb_stream_name = rgb_stream_name
        self._pointcloud_stream_name = pointcloud_stream_name

        # Create some nodes
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        pointcloud_node: dai.node.PointCloud = pipeline.create(dai.node.PointCloud)
        sync_node = pipeline.create(dai.node.Sync)

        # Output XLink
        x_out = pipeline.create(dai.node.XLinkOut)
        x_out.input.setBlocking(False)

        # Setup RGB camera
        cam_rgb.setResolution(
            resolution=dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        cam_rgb.setBoardSocket(boardSocket=dai.CameraBoardSocket.CAM_A)
        cam_rgb.setIspScale(1, 3)
        cam_rgb.setColorOrder(colorOrder=dai.ColorCameraProperties.ColorOrder.RGB)

        # Setup mono cameras
        mono_left.setResolution(
            resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        mono_left.setCamera(name="left")
        mono_right.setResolution(
            resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        mono_right.setCamera(name="right")

        # Setup stereo depth
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(median=dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(enable=True)
        depth.setExtendedDisparity(enable=False)
        depth.setSubpixel(enable=True)
        depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        config = depth.initialConfig.get()
        config.postProcessing.thresholdFilter.minRange = int(
            min_depth * 1000
        )  # Convert to mm
        config.postProcessing.thresholdFilter.maxRange = int(
            max_depth * 1000
        )  # Convert to mm
        depth.initialConfig.set(config)

        # Setup point cloud node
        pointcloud_node.initialConfig.setSparse(False)

        # Link the nodes
        mono_left.out.link(depth.left)
        mono_right.out.link(depth.right)
        depth.depth.link(pointcloud_node.inputDepth)
        cam_rgb.isp.link(sync_node.inputs[rgb_stream_name])
        pointcloud_node.outputPointCloud.link(sync_node.inputs[pointcloud_stream_name])
        sync_node.out.link(x_out.input)
        x_out.setStreamName(streamName="out")

        # Setup pointcloud input config link
        config_input = pipeline.create(dai.node.XLinkIn)
        config_input.setStreamName(streamName="config")
        config_input.out.link(pointcloud_node.inputConfig)

    def setup_with_device(self, device: dai.Device) -> None:
        self._device = device
        self._pcl_config_input = device.getInputQueue(
            name="config",
            maxSize=4,  # type: ignore
            blocking=False,  # type: ignore
        )
        self._queue = device.getOutputQueue(name="out", maxSize=4, blocking=False)  # type: ignore

    def read_img_and_pointcloud(
        self,
        cam_to_world_transform: npt.NDArray[np.float32] = np.eye(4, dtype=np.float32),
    ) -> tuple[Image, Pointcloud]:
        if not self._device.isPipelineRunning():
            raise RuntimeError("Pipeline is not running. Call setup_with_device first.")

        in_msg = self._queue.get()
        in_rgb = in_msg[self._rgb_stream_name]  # type: ignore
        in_pcl = in_msg[self._pointcloud_stream_name]  # type: ignore

        # Get the color RGB frame
        rgb_frame = in_rgb.getCvFrame()

        # Get the points
        if in_pcl is None:
            raise RuntimeError("Point cloud data is not available.")
        points = in_pcl.getPoints().astype(np.float32) * 1e-3  # Convert to meters

        # Create pointcloud transformation config and send it
        pointcloud_config = dai.PointCloudConfig()
        pointcloud_config.setTransformationMatrix(cam_to_world_transform)
        pointcloud_config.setSparse(False)
        self._pcl_config_input.send(pointcloud_config)  # type: ignore

        return rgb_frame, points
