import os
import cv2
import time
import httpx
import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from app.schemas.snapshot import CameraTypeEnum

logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """Result returned after attempting to capture a frame from a CCTV camera."""
    success: bool
    image_bytes: Optional[bytes] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error_message: Optional[str] = None
    captured_at: datetime = datetime.utcnow()
    protocol: Optional[str] = None


class BaseCameraAdapter:
    """Abstract base adapter for camera protocol implementations."""
    async def capture(self, **kwargs) -> CaptureResult:
        raise NotImplementedError


class RTSPCameraAdapter(BaseCameraAdapter):
    """
    RTSP Stream Adapter.
    Uses OpenCV with FFmpeg backend over TCP for reliable frame retrieval.
    Works with Hikvision, Dahua, Uniview, Tiandy, Axis, Hanwha and generic RTSP IP cameras.
    """

    def __init__(self, timeout_seconds: float = 8.0):
        self.timeout_seconds = timeout_seconds

    def _format_rtsp_url(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 554
    ) -> str:
        """Embeds credentials and standardizes the RTSP URL if provided separately."""
        if not stream_url.startswith("rtsp://"):
            stream_url = f"rtsp://{stream_url}"

        parsed = urlparse(stream_url)
        
        # Inject username/password into the URL netloc if not already present
        if username and password and "@" not in parsed.netloc:
            host_port = parsed.netloc
            new_netloc = f"{username}:{password}@{host_port}"
            parsed = parsed._replace(netloc=new_netloc)

        return urlunparse(parsed)

    def _sync_capture(self, url: str) -> CaptureResult:
        """Synchronously connects to RTSP stream and captures the latest frame."""
        start_time = time.time()
        
        # Set environment options for FFmpeg inside OpenCV
        # Force TCP transport for RTSP to prevent UDP packet loss and gray frames
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000"

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                return CaptureResult(
                    success=False,
                    error_message=f"Failed to connect to RTSP stream at {url}",
                    protocol="RTSP"
                )

            # Grab a few frames to flush any stale buffer and get the most recent frame
            ret = False
            frame = None
            for _ in range(3):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            if not ret or frame is None:
                return CaptureResult(
                    success=False,
                    error_message="Connected to RTSP stream but could not read video frame",
                    protocol="RTSP"
                )

            # Encode frame to JPEG format (quality 92)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
            success, buffer = cv2.imencode(".jpg", frame, encode_param)
            if not success:
                return CaptureResult(
                    success=False,
                    error_message="Failed to encode captured frame to JPEG format",
                    protocol="RTSP"
                )

            height, width = frame.shape[:2]
            elapsed = time.time() - start_time
            logger.info(f"Successfully captured RTSP frame ({width}x{height}) in {elapsed:.2f}s from {url}")

            return CaptureResult(
                success=True,
                image_bytes=buffer.tobytes(),
                width=width,
                height=height,
                protocol="RTSP"
            )

        except Exception as e:
            logger.error(f"Error during RTSP capture: {e}")
            return CaptureResult(
                success=False,
                error_message=str(e),
                protocol="RTSP"
            )
        finally:
            cap.release()

    async def capture(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 554,
        **kwargs
    ) -> CaptureResult:
        """Asynchronously captures a frame by running OpenCV in a worker thread."""
        formatted_url = self._format_rtsp_url(
            stream_url=stream_url,
            username=username,
            password=password,
            ip_address=ip_address,
            port=port
        )
        try:
            # Run blocking OpenCV call in async threadpool with timeout
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_capture, formatted_url),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"RTSP capture timed out after {self.timeout_seconds} seconds")
            return CaptureResult(
                success=False,
                error_message=f"RTSP capture timed out after {self.timeout_seconds}s",
                protocol="RTSP"
            )


class HTTPSnapshotCameraAdapter(BaseCameraAdapter):
    """
    HTTP/HTTPS Snapshot API Adapter.
    Directly fetches instant JPEG snapshots from camera HTTP endpoints.
    Supported Vendors:
    - Hikvision: http://<ip>/ISAPI/Streaming/channels/101/picture
    - Dahua: http://<ip>/cgi-bin/snapshot.cgi?channel=1
    - Uniview: http://<ip>/LAPI/V1.0/Channels/1/Media/Snapshot
    - Generic HTTP Snapshot URLs
    """

    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds

    async def capture(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 80,
        **kwargs
    ) -> CaptureResult:
        start_time = time.time()
        url = stream_url

        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"

        # Setup authentication if credentials provided
        auth = None
        if username and password:
            # We first try DigestAuth, if fallback needed BasicAuth is used
            auth = httpx.DigestAuth(username, password)

        async with httpx.AsyncClient(timeout=self.timeout_seconds, verify=False) as client:
            try:
                response = await client.get(url, auth=auth)
                
                # If Digest auth failed with 401, retry with standard Basic auth
                if response.status_code == 401 and username and password:
                    response = await client.get(url, auth=httpx.BasicAuth(username, password))

                if response.status_code != 200:
                    return CaptureResult(
                        success=False,
                        error_message=f"HTTP Snapshot returned status {response.status_code}: {response.text[:100]}",
                        protocol="HTTP_SNAPSHOT"
                    )

                image_bytes = response.content
                
                # Decode image size using OpenCV or numpy
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                width, height = (None, None)
                if img is not None:
                    height, width = img.shape[:2]

                elapsed = time.time() - start_time
                logger.info(f"Successfully fetched HTTP snapshot ({width}x{height}) in {elapsed:.2f}s from {url}")

                return CaptureResult(
                    success=True,
                    image_bytes=image_bytes,
                    width=width,
                    height=height,
                    protocol="HTTP_SNAPSHOT"
                )

            except Exception as e:
                logger.error(f"Error fetching HTTP snapshot from {url}: {e}")
                return CaptureResult(
                    success=False,
                    error_message=str(e),
                    protocol="HTTP_SNAPSHOT"
                )


class ONVIFCameraAdapter(BaseCameraAdapter):
    """
    ONVIF Protocol Adapter.
    Fetches snapshot URI and retrieves snapshot image via standardized ONVIF interface.
    """

    def __init__(self, timeout_seconds: float = 8.0):
        self.timeout_seconds = timeout_seconds
        self._http_adapter = HTTPSnapshotCameraAdapter(timeout_seconds=timeout_seconds)

    async def capture(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 80,
        **kwargs
    ) -> CaptureResult:
        # If the URL is already an ONVIF media snapshot URI, pass to HTTP adapter
        if stream_url.startswith("http://") or stream_url.startswith("https://"):
            return await self._http_adapter.capture(
                stream_url=stream_url,
                username=username,
                password=password,
                ip_address=ip_address,
                port=port
            )
        
        # Fallback to RTSP adapter if stream_url is RTSP format
        rtsp_adapter = RTSPCameraAdapter(timeout_seconds=self.timeout_seconds)
        return await rtsp_adapter.capture(
            stream_url=stream_url,
            username=username,
            password=password,
            ip_address=ip_address,
            port=port
        )


class MockCameraGenerator:
    """
    Simulated camera frame generator for testing/development when real cameras are not connected.
    Creates a realistic mock image of a charging station with a car and license plate watermark.
    """

    @staticmethod
    def generate_mock_frame(cs_id: str, cp_id: str, plate_number: Optional[str] = None) -> CaptureResult:
        width, height = 1280, 720
        # Create dark outdoor background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (35, 35, 40) # Dark gray asphalt background

        # Draw a simulated EV charging station bay
        cv2.rectangle(image, (150, 100), (1130, 650), (60, 60, 65), -1)
        cv2.rectangle(image, (150, 100), (1130, 650), (0, 200, 100), 4) # Green parking border

        # Draw simulated car body
        cv2.rectangle(image, (300, 220), (980, 520), (180, 120, 50), -1) # Blueish vehicle body
        cv2.rectangle(image, (420, 240), (860, 360), (70, 70, 70), -1)   # Windshield

        # Draw simulated license plate
        sample_plate = plate_number or "01A777AA"
        cv2.rectangle(image, (520, 430), (760, 490), (255, 255, 255), -1) # White plate background
        cv2.rectangle(image, (520, 430), (760, 490), (0, 0, 0), 2)
        cv2.putText(image, sample_plate, (540, 475), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # Draw OSD / Watermark header
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        osd_text = f"CCTV CAM - STATION: {cs_id} | CP: {cp_id} | {timestamp_str}"
        cv2.putText(image, osd_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode(".jpg", image, encode_param)

        return CaptureResult(
            success=True,
            image_bytes=buffer.tobytes(),
            width=width,
            height=height,
            captured_at=datetime.utcnow(),
            protocol="SIMULATED_MOCK"
        )


class CameraService:
    """
    Universal Camera Manager.
    Automatically routes requests to the proper adapter (RTSP, HTTP_SNAPSHOT, ONVIF)
    with graceful error handling and optional mock fallback for development.
    """

    def __init__(self, enable_mock_fallback: bool = True):
        self.adapters = {
            CameraTypeEnum.RTSP: RTSPCameraAdapter(),
            CameraTypeEnum.HTTP_SNAPSHOT: HTTPSnapshotCameraAdapter(),
            CameraTypeEnum.ONVIF: ONVIFCameraAdapter(),
        }
        self.enable_mock_fallback = enable_mock_fallback

    async def capture_snapshot(
        self,
        camera_type: CameraTypeEnum,
        stream_url: str,
        cs_id: str = "DEFAULT_CS",
        cp_id: str = "DEFAULT_CP",
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = None,
        expected_plate_number: Optional[str] = None,
    ) -> CaptureResult:
        """
        Captures a frame using the specified camera protocol.
        If the camera is unreachable and fallback is enabled, returns a test/mock frame.
        """
        adapter = self.adapters.get(camera_type, self.adapters[CameraTypeEnum.RTSP])

        logger.info(f"Initiating snapshot capture using protocol: {camera_type} for Station {cs_id}/{cp_id} from {stream_url}")

        result = await adapter.capture(
            stream_url=stream_url,
            username=username,
            password=password,
            ip_address=ip_address,
            port=port
        )

        # If live capture failed and fallback is enabled (e.g. during local tests without active camera)
        if not result.success and self.enable_mock_fallback:
            logger.warning(
                f"Live camera capture failed ({result.error_message}). Using mock generator fallback."
            )
            mock_result = MockCameraGenerator.generate_mock_frame(
                cs_id=cs_id,
                cp_id=cp_id,
                plate_number=expected_plate_number
            )
            mock_result.error_message = f"Simulated Fallback (Live Camera Error: {result.error_message})"
            return mock_result

        return result


# Singleton instance
camera_service = CameraService(enable_mock_fallback=True)
