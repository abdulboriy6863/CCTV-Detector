import os
import cv2
import time
import httpx
import asyncio
import logging
import threading
import urllib.parse
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

from app.core.config import get_kst_now
from app.schemas.snapshot import CameraTypeEnum

logger = logging.getLogger(__name__)


# =====================================================================
# CCTV Brand Definitions & Stream Path Presets
# =====================================================================
VENDOR_RTSP_PRESETS = {
    "hikvision": [
        "/Streaming/Channels/101",               # Hikvision Main Stream (HD)
        "/Streaming/Channels/102",               # Hikvision Sub Stream
        "/ISAPI/Streaming/channels/101",
        "/ISAPI/Streaming/channels/102",
        "/h264/ch1/main/av_stream",
        "/h264/ch1/sub/av_stream",
    ],
    "dahua": [
        "/cam/realmonitor?channel=1&subtype=0",  # Dahua / Imou Main Stream
        "/cam/realmonitor?channel=1&subtype=1",  # Dahua / Imou Sub Stream
        "/cam/realmonitor?channel=0&subtype=0",
        "/live",
    ],
    "hanwha": [
        "/profile2/media.smp",                   # Hanwha Techwin / Wisenet Profile 2 (Recommended for LPR)
        "/profile1/media.smp",                   # Hanwha Profile 1
        "/stw-cgi/video.cgi?msubmenu=stream&action=view&profile=2",
        "/onvif-media/media.amp?profile=profile_1",
    ],
    "uniview": [
        "/unicast/c1/s0/live",                   # Uniview (UNV) Main
        "/unicast/c1/s1/live",                   # Uniview (UNV) Sub
        "/media/video1",
        "/media/video2",
    ],
    "axis": [
        "/axis-media/media.amp",                 # Axis Main
        "/axis-media/media.3gp",
        "/onvif-media/media.amp",
    ],
    "tplink": [
        "/stream1",                              # TP-Link Tapo / Vigi Main
        "/stream2",                              # TP-Link Tapo / Vigi Sub
    ],
    "tiandy": [
        "/1/1",
        "/1/2",
    ],
    "xmeye": [
        "/user={user}&password={pass}&channel=1&stream=0.sdp",
        "/h264Preview_01_main",
        "/h264Preview_01_sub",
        "/ch01.264",
        "/ch0_0.264",
        "/live/ch0",
        "/live/ch1",
    ],
    "generic": [
        "/stream1",
        "/stream2",
        "/live/ch0",
        "/live/ch1",
        "/video1",
        "/video2",
        "/onvif1",
        "/onvif2",
        "/h264",
        "/h265",
        "/",
    ]
}

VENDOR_HTTP_SNAPSHOT_PRESETS = {
    "hikvision": [
        "/ISAPI/Streaming/channels/101/picture",
        "/Streaming/channels/1/picture",
        "/Streaming/channels/101/picture",
    ],
    "dahua": [
        "/cgi-bin/snapshot.cgi?channel=1",
        "/cgi-bin/snapshot.cgi",
        "/onvif/snapshot",
    ],
    "hanwha": [
        "/stw-cgi/video.cgi?msubmenu=snapshot&action=view",
        "/cgi-bin/camera?resolution=1920x1080",
    ],
    "uniview": [
        "/LAPI/V1.0/Channels/1/Media/Snapshot",
        "/Images/Snapshot",
    ],
    "axis": [
        "/axis-cgi/jpg/image.cgi",
        "/axis-cgi/bitmap/image.bmp",
    ],
    "generic": [
        "/snapshot.jpg",
        "/image.jpg",
        "/snap.jpg",
        "/api/snapshot",
        "/tmpfs/auto.jpg",
        "/jpeg",
    ]
}


class RTSPStreamHub:
    """
    Zero-latency RTSP Live Stream Hub with self-healing reconnection.
    Maintains a dedicated background worker thread per camera URL that constantly
    drains the OpenCV/FFmpeg socket buffer at full camera frame rate and caches only the latest JPEG.
    """
    _instances: Dict[str, "RTSPStreamHub"] = {}
    _lock = threading.Lock()

    def __init__(self, url: str):
        self.url = url
        self.latest_jpeg: Optional[bytes] = None
        self.last_frame_time: float = 0.0
        self.subscribers: int = 0
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    @classmethod
    def get_stream(cls, url: str) -> "RTSPStreamHub":
        with cls._lock:
            if url not in cls._instances:
                cls._instances[url] = RTSPStreamHub(url)
            return cls._instances[url]

    def add_subscriber(self):
        with self._lock:
            self.subscribers += 1
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._reader_worker, daemon=True)
                self.thread.start()

    def remove_subscriber(self):
        with self._lock:
            self.subscribers = max(0, self.subscribers - 1)
            if self.subscribers == 0:
                self.running = False

    def _reader_worker(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|buffer_size;1024000|max_delay;500000"
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        target_width = 960

        try:
            while self.running:
                if not cap.isOpened():
                    cap.open(self.url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        time.sleep(1.0)
                        continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                if frame.shape[1] > target_width:
                    scale = target_width / frame.shape[1]
                    new_h = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)

                success, buffer = cv2.imencode(".jpg", frame, encode_param)
                if success:
                    self.latest_jpeg = buffer.tobytes()
                    self.last_frame_time = time.time()
        except Exception as e:
            logger.error(f"Error in RTSPStreamHub worker for {self.url}: {e}")
        finally:
            cap.release()


@dataclass
class CaptureResult:
    """Result returned after attempting to capture a frame from a CCTV camera."""
    success: bool
    image_bytes: Optional[bytes] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error_message: Optional[str] = None
    captured_at: datetime = field(default_factory=get_kst_now)
    protocol: Optional[str] = None


class BaseCameraAdapter:
    """Abstract base adapter for camera protocol implementations."""
    async def capture(self, **kwargs) -> CaptureResult:
        raise NotImplementedError


class RTSPCameraAdapter(BaseCameraAdapter):
    """
    RTSP Stream Adapter.
    Uses OpenCV with FFmpeg backend over TCP/UDP for reliable multi-vendor frame retrieval.
    Works seamlessly with Hikvision, Dahua, Uniview, Tiandy, Axis, Hanwha and generic RTSP IP cameras.
    """

    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def format_rtsp_url(
        stream_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 554
    ) -> str:
        """
        Robustly constructs and standardizes RTSP URLs with credential escaping.
        Handles missing IPs, existing credentials, custom ports, and path preservation.
        """
        raw_url = (stream_url or "").strip()
        eff_port = port or 554

        # If stream_url is completely empty but IP is provided
        if not raw_url and ip_address:
            raw_url = f"rtsp://{ip_address.strip()}:{eff_port}/stream1"

        if raw_url and not (raw_url.startswith("rtsp://") or raw_url.startswith("rtsps://")):
            raw_url = f"rtsp://{raw_url}"

        parsed = urlparse(raw_url)

        # Extract or resolve host and port
        host = parsed.hostname or (ip_address.strip() if ip_address else "127.0.0.1")
        target_port = parsed.port or eff_port

        # Resolve credentials
        user = username.strip() if username else parsed.username
        pwd = password.strip() if password else parsed.password

        # Rebuild netloc with URL-encoded credentials
        if user and pwd:
            user_enc = urllib.parse.quote(user, safe="")
            pass_enc = urllib.parse.quote(pwd, safe="")
            netloc = f"{user_enc}:{pass_enc}@{host}"
        elif user:
            user_enc = urllib.parse.quote(user, safe="")
            netloc = f"{user_enc}@{host}"
        else:
            netloc = f"{host}"

        if target_port:
            netloc = f"{netloc}:{target_port}"

        # Preserve path and query params (e.g. /cam/realmonitor?channel=1&subtype=0)
        path = parsed.path or ""
        if not path and not parsed.query:
            path = "/stream1"

        new_parsed = parsed._replace(
            scheme="rtsp",
            netloc=netloc,
            path=path
        )
        return urlunparse(new_parsed)

    def _sync_capture(self, url: str, timeout_seconds: float = 4.0, use_tcp: bool = True) -> CaptureResult:
        """Synchronously connects to RTSP stream and captures the latest frame."""
        start_time = time.time()
        
        transport = "tcp" if use_tcp else "udp"
        stimeout_us = max(1000000, int(timeout_seconds * 1_000_000))
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|stimeout;{stimeout_us}|buffer_size;1024000|max_delay;500000"
        )

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                return CaptureResult(
                    success=False,
                    error_message=f"RTSP stream connection failed ({transport}) at {url}",
                    protocol="RTSP"
                )

            # Read decoded frame with retry for keyframe/stream synchronization (up to 8 attempts)
            ret = False
            frame = None
            for _ in range(8):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    break
                time.sleep(0.04)

            if not ret or frame is None:
                return CaptureResult(
                    success=False,
                    error_message="Connected to RTSP stream but could not decode video frame",
                    protocol="RTSP"
                )

            # Encode frame to JPEG format (quality 92)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
            success, buffer = cv2.imencode(".jpg", frame, encode_param)
            if not success:
                return CaptureResult(
                    success=False,
                    error_message="Failed to encode captured frame to JPEG",
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
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> CaptureResult:
        """Asynchronously captures a frame with automatic TCP->UDP fallback."""
        eff_timeout = timeout_seconds or kwargs.get('timeout_seconds', self.timeout_seconds)
        formatted_url = self.format_rtsp_url(
            stream_url=stream_url,
            username=username,
            password=password,
            ip_address=ip_address,
            port=port
        )
        try:
            # 1. Try TCP transport first
            res = await asyncio.wait_for(
                asyncio.to_thread(self._sync_capture, formatted_url, eff_timeout, True),
                timeout=eff_timeout
            )
            if res.success:
                return res

            # 2. Fallback to UDP if TCP failed
            logger.debug(f"TCP capture failed for {formatted_url}, trying UDP transport...")
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_capture, formatted_url, eff_timeout * 0.8, False),
                timeout=eff_timeout
            )

        except asyncio.TimeoutError:
            logger.error(f"RTSP capture timed out after {eff_timeout} seconds")
            return CaptureResult(
                success=False,
                error_message=f"RTSP capture timed out after {eff_timeout}s",
                protocol="RTSP"
            )

    async def stream_frames(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 554,
        target_fps: int = 20,
        **kwargs
    ):
        """
        Asynchronously yields multipart MJPEG video frames for continuous live browser preview.
        Uses RTSPStreamHub to completely eliminate browser buffering delay.
        """
        formatted_url = self.format_rtsp_url(
            stream_url=stream_url,
            username=username,
            password=password,
            ip_address=ip_address,
            port=port
        )
        hub = RTSPStreamHub.get_stream(formatted_url)
        hub.add_subscriber()
        delay = 1.0 / target_fps
        last_sent_time = 0.0

        try:
            while True:
                if hub.latest_jpeg and hub.last_frame_time != last_sent_time:
                    last_sent_time = hub.last_frame_time
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + hub.latest_jpeg + b"\r\n")
                elif not hub.latest_jpeg:
                    mock = MockCameraGenerator.generate_mock_frame(cs_id="LIVE", cp_id="CAM")
                    if mock.image_bytes:
                        yield (b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n" + mock.image_bytes + b"\r\n")
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass
        finally:
            hub.remove_subscriber()


class HTTPSnapshotCameraAdapter(BaseCameraAdapter):
    """
    HTTP/HTTPS Snapshot API Adapter.
    Directly fetches instant JPEG snapshots from camera HTTP/HTTPS endpoints.
    Supported Vendors:
    - Hikvision: http://<ip>/ISAPI/Streaming/channels/101/picture
    - Dahua: http://<ip>/cgi-bin/snapshot.cgi?channel=1
    - Uniview: http://<ip>/LAPI/V1.0/Channels/1/Media/Snapshot
    - Hanwha: http://<ip>/stw-cgi/video.cgi?msubmenu=snapshot&action=view
    - Axis: http://<ip>/axis-cgi/jpg/image.cgi
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def format_http_url(
        stream_url: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 80
    ) -> str:
        raw_url = (stream_url or "").strip()
        eff_port = port or 80
        if eff_port == 554:
            eff_port = 80

        if not raw_url and ip_address:
            scheme = "https" if eff_port == 443 else "http"
            port_str = f":{eff_port}" if eff_port not in [80, 443] else ""
            return f"{scheme}://{ip_address.strip()}{port_str}/snapshot.jpg"

        if raw_url.startswith("/"):
            # Path only provided
            scheme = "https" if eff_port == 443 else "http"
            host = ip_address.strip() if ip_address else "127.0.0.1"
            port_str = f":{eff_port}" if eff_port not in [80, 443] else ""
            return f"{scheme}://{host}{port_str}{raw_url}"

        if not (raw_url.startswith("http://") or raw_url.startswith("https://")):
            raw_url = f"http://{raw_url}"

        parsed = urlparse(raw_url)
        host = parsed.hostname or (ip_address.strip() if ip_address else "127.0.0.1")
        target_port = parsed.port or eff_port
        if target_port == 554:
            target_port = 80

        port_str = f":{target_port}" if target_port not in [80, 443] else ""
        netloc = f"{host}{port_str}"
        path = parsed.path or "/snapshot.jpg"

        new_parsed = parsed._replace(
            scheme=parsed.scheme or ("https" if target_port == 443 else "http"),
            netloc=netloc,
            path=path
        )
        return urlunparse(new_parsed)

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
        url = self.format_http_url(stream_url=stream_url, ip_address=ip_address, port=port)

        auth = None
        if username and password:
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

    async def stream_frames(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 80,
        target_fps: int = 5,
        **kwargs
    ):
        delay = 1.0 / target_fps
        try:
            while True:
                res = await self.capture(
                    stream_url=stream_url,
                    username=username,
                    password=password,
                    ip_address=ip_address,
                    port=port
                )
                if res.success and res.image_bytes:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + res.image_bytes + b"\r\n")
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass


class ONVIFCameraAdapter(BaseCameraAdapter):
    """
    ONVIF Protocol Adapter.
    Fetches snapshot URI and retrieves snapshot image via standardized ONVIF interface.
    """

    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds
        self._http_adapter = HTTPSnapshotCameraAdapter(timeout_seconds=timeout_seconds)
        self._rtsp_adapter = RTSPCameraAdapter(timeout_seconds=timeout_seconds)

    async def capture(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 80,
        **kwargs
    ) -> CaptureResult:
        if stream_url and (stream_url.startswith("http://") or stream_url.startswith("https://")):
            return await self._http_adapter.capture(
                stream_url=stream_url,
                username=username,
                password=password,
                ip_address=ip_address,
                port=port
            )
        return await self._rtsp_adapter.capture(
            stream_url=stream_url,
            username=username,
            password=password,
            ip_address=ip_address,
            port=port
        )

    async def stream_frames(
        self,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = 80,
        **kwargs
    ):
        if stream_url and (stream_url.startswith("http://") or stream_url.startswith("https://")):
            async for chunk in self._http_adapter.stream_frames(
                stream_url=stream_url, username=username, password=password, ip_address=ip_address, port=port
            ):
                yield chunk
        else:
            async for chunk in self._rtsp_adapter.stream_frames(
                stream_url=stream_url, username=username, password=password, ip_address=ip_address, port=port
            ):
                yield chunk


class MockCameraGenerator:
    """
    Simulated camera frame generator for unit testing and offline development.
    Uses authentic Korean license plates for test compliance.
    """

    @staticmethod
    def generate_mock_frame(cs_id: str, cp_id: str, plate_number: Optional[str] = None) -> CaptureResult:
        width, height = 1280, 720
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (35, 35, 40)

        # Simulated EV charging bay
        cv2.rectangle(image, (150, 100), (1130, 650), (60, 60, 65), -1)
        cv2.rectangle(image, (150, 100), (1130, 650), (0, 200, 100), 4)

        # Vehicle body
        cv2.rectangle(image, (300, 220), (980, 520), (180, 120, 50), -1)
        cv2.rectangle(image, (420, 240), (860, 360), (70, 70, 70), -1)

        # Legal Korean License Plate background
        sample_plate = plate_number or "81머 2072"
        cv2.rectangle(image, (520, 430), (760, 490), (225, 200, 130), -1) # Light blue EV plate background
        cv2.rectangle(image, (520, 430), (760, 490), (0, 0, 0), 2)
        cv2.putText(image, sample_plate, (535, 475), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

        # Header watermark
        timestamp_str = get_kst_now().strftime("%Y-%m-%d %H:%M:%S")
        osd_text = f"CCTV CAM - STATION: {cs_id} | CP: {cp_id} | {timestamp_str}"
        cv2.putText(image, osd_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode(".jpg", image, encode_param)

        return CaptureResult(
            success=True,
            image_bytes=buffer.tobytes(),
            width=width,
            height=height,
            captured_at=get_kst_now(),
            protocol="SIMULATED_MOCK"
        )


class CameraService:
    """
    Central CCTV Camera Service with unified multi-vendor adapter management,
    instant automatic protocol & stream discovery, live frame streaming, and error handling.
    """

    def __init__(self, enable_mock_fallback: bool = False):
        self.adapters = {
            CameraTypeEnum.RTSP: RTSPCameraAdapter(timeout_seconds=5.0),
            CameraTypeEnum.HTTP_SNAPSHOT: HTTPSnapshotCameraAdapter(timeout_seconds=4.0),
            CameraTypeEnum.ONVIF: ONVIFCameraAdapter(timeout_seconds=5.0),
        }
        self.enable_mock_fallback = enable_mock_fallback

    async def probe_camera(
        self,
        ip_address: Optional[str] = None,
        port: Optional[int] = None,
        camera_type: Optional[CameraTypeEnum] = None,
        stream_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        brand: Optional[str] = None,
        timeout_seconds: float = 4.0
    ) -> Tuple[bool, str, Optional[str], Optional[CameraTypeEnum], Optional[CaptureResult]]:
        """
        Fast Smart Multi-Vendor Camera Probe:
        1. Tests explicit stream_url if provided.
        2. Parallel port scan (554, 80, 8554, 8080, 443, 8899, 37777, 8000).
        3. Probes vendor-specific presets (Hikvision, Dahua, Hanwha, Uniview, Axis, TP-Link, etc.).
        4. Returns working URL, detected protocol, and verified live snapshot frame.
        """
        import socket

        # 1. If explicit stream_url provided, test it first
        if stream_url and "://" in stream_url:
            parsed = urlparse(stream_url)
            detected_type = camera_type or (
                CameraTypeEnum.HTTP_SNAPSHOT if parsed.scheme in ["http", "https"] else CameraTypeEnum.RTSP
            )
            target_ip = parsed.hostname or (ip_address.strip() if ip_address else None)
            target_port = parsed.port or port or (554 if detected_type == CameraTypeEnum.RTSP else 80)

            adapter = self.adapters.get(detected_type, self.adapters[CameraTypeEnum.RTSP])
            try:
                res = await adapter.capture(
                    stream_url=stream_url,
                    username=username,
                    password=password,
                    ip_address=target_ip,
                    port=target_port,
                    timeout_seconds=2.5
                )
                if res.success:
                    return True, "카메라 연결 성공", stream_url, detected_type, res
            except Exception as e:
                logger.debug(f"Direct stream_url probe failed: {e}")

        target_ip = ip_address.strip() if ip_address else None
        if not target_ip and stream_url:
            parsed = urlparse(stream_url if "://" in stream_url else f"rtsp://{stream_url}")
            target_ip = parsed.hostname

        if not target_ip:
            return False, "IP 주소를 입력해주세요.", None, None, None

        # 2. Parallel fast TCP port check (0.35s timeout)
        def _check_tcp_port(p: int, tout: float = 0.35) -> bool:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(tout)
                s.connect((target_ip, p))
                s.close()
                return True
            except Exception:
                return False

        candidate_ports = []
        if port:
            candidate_ports.append(port)
        for p in [554, 80, 8554, 8080, 443, 8899, 37777, 8000]:
            if p not in candidate_ports:
                candidate_ports.append(p)

        port_results = await asyncio.gather(
            *[asyncio.to_thread(_check_tcp_port, p, 0.35) for p in candidate_ports]
        )
        open_ports = [p for p, is_op in zip(candidate_ports, port_results) if is_op]

        if not open_ports:
            # If standard scan found no open ports, still try 554 as default RTSP
            open_ports = [port or 554]

        # 3. Build prioritized list of RTSP candidate paths
        rtsp_paths = []
        if brand and brand.lower() in VENDOR_RTSP_PRESETS:
            rtsp_paths.extend(VENDOR_RTSP_PRESETS[brand.lower()])

        # Add Hanwha, Hikvision, Dahua, Uniview, TP-Link and Generic presets
        for vendor_key in ["hanwha", "hikvision", "dahua", "uniview", "tplink", "generic"]:
            if brand and brand.lower() == vendor_key:
                continue
            for p in VENDOR_RTSP_PRESETS.get(vendor_key, []):
                if p not in rtsp_paths:
                    rtsp_paths.append(p)

        if stream_url and "://" in stream_url:
            custom_path = urlparse(stream_url).path
            if custom_path and custom_path not in rtsp_paths:
                rtsp_paths.insert(0, custom_path)

        # 4. Probe RTSP candidate paths
        rtsp_ports = [p for p in open_ports if p in [554, 8554, 5540, 37777, 8000] or (camera_type == CameraTypeEnum.RTSP and p == port)]
        if not rtsp_ports:
            rtsp_ports = [554]

        rtsp_adapter = self.adapters[CameraTypeEnum.RTSP]
        rtsp_p = rtsp_ports[0]

        for path in rtsp_paths[:12]:
            test_url = f"rtsp://{target_ip}:{rtsp_p}{path}"
            try:
                res = await rtsp_adapter.capture(
                    stream_url=test_url,
                    username=username,
                    password=password,
                    ip_address=target_ip,
                    port=rtsp_p,
                    timeout_seconds=2.0
                )
                if res.success:
                    return True, f"RTSP 연결 성공 ({path})", test_url, CameraTypeEnum.RTSP, res
            except Exception as e:
                logger.debug(f"RTSP probe failed for {test_url}: {e}")

        # 5. Probe HTTP Snapshots if HTTP_SNAPSHOT requested or ports 80/443/8080 open
        http_ports = [p for p in open_ports if p in [80, 443, 8080]]
        if http_ports or camera_type == CameraTypeEnum.HTTP_SNAPSHOT:
            hp = http_ports[0] if http_ports else (port or 80)
            http_paths = []
            if brand and brand.lower() in VENDOR_HTTP_SNAPSHOT_PRESETS:
                http_paths.extend(VENDOR_HTTP_SNAPSHOT_PRESETS[brand.lower()])
            for vendor_key in ["hanwha", "hikvision", "dahua", "uniview", "generic"]:
                for p in VENDOR_HTTP_SNAPSHOT_PRESETS.get(vendor_key, []):
                    if p not in http_paths:
                        http_paths.append(p)

            http_adapter = self.adapters[CameraTypeEnum.HTTP_SNAPSHOT]
            scheme = "https" if hp == 443 else "http"
            port_str = f":{hp}" if hp not in [80, 443] else ""

            for spath in http_paths[:8]:
                test_url = f"{scheme}://{target_ip}{port_str}{spath}"
                try:
                    res = await http_adapter.capture(
                        stream_url=test_url,
                        username=username,
                        password=password,
                        ip_address=target_ip,
                        port=hp,
                        timeout_seconds=1.5
                    )
                    if res.success:
                        return True, f"HTTP Snapshot 연결 성공 ({spath})", test_url, CameraTypeEnum.HTTP_SNAPSHOT, res
                except Exception as e:
                    logger.debug(f"HTTP probe failed for {test_url}: {e}")

        # 6. Fallback failure message
        return (
            False,
            f"IP {target_ip}:{port or open_ports[0]} 카메라에 연결할 수 없습니다. (IP, 포트, 인증 정보 및 RTSP 스트림 경로를 확인해주세요)",
            None,
            None,
            None
        )

    async def check_camera_reachability(
        self,
        camera_type: CameraTypeEnum,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = None,
        brand: Optional[str] = None,
        timeout_seconds: float = 4.0
    ) -> Tuple[bool, str, Optional[str], Optional[CameraTypeEnum]]:
        """
        Tests if a camera is reachable and returns (is_ok, message, working_url, detected_type).
        """
        success, msg, working_url, detected_type, _ = await self.probe_camera(
            ip_address=ip_address,
            port=port,
            camera_type=camera_type,
            stream_url=stream_url,
            username=username,
            password=password,
            brand=brand,
            timeout_seconds=timeout_seconds
        )
        return success, msg, working_url or stream_url, detected_type or camera_type

    _snapshot_cache: Dict[str, Tuple[float, CaptureResult]] = {}
    _last_known_good: Dict[str, Tuple[float, CaptureResult]] = {}
    _cache_lock = threading.Lock()
    _ip_locks: Dict[str, asyncio.Lock] = {}
    _ip_locks_guard = threading.Lock()

    def _get_cache_key(
        self,
        camera_type: CameraTypeEnum,
        stream_url: str,
        ip_address: Optional[str] = None,
        port: Optional[int] = None
    ) -> str:
        clean_url = (stream_url or "").strip()
        if "@" in clean_url:
            parts = clean_url.split("@", 1)
            prefix = parts[0].split("://")[0] + "://"
            clean_url = prefix + parts[1]
        host = (ip_address or "").strip()
        if not host and clean_url:
            try:
                host = clean_url.split("://")[-1].split("/")[0].split(":")[0]
            except Exception:
                host = "unknown"
        eff_port = port or (80 if camera_type == CameraTypeEnum.HTTP_SNAPSHOT else 554)
        return f"{camera_type.value}_{host}_{eff_port}_{clean_url}"

    def _get_ip_lock(self, ip: str) -> asyncio.Lock:
        with self._ip_locks_guard:
            if ip not in self._ip_locks:
                self._ip_locks[ip] = asyncio.Lock()
            return self._ip_locks[ip]

    @staticmethod
    def generate_standby_frame(camera_title: str = "CCTV Camera") -> bytes:
        """Generates a high-quality CCTV standby placeholder when connecting."""
        import numpy as np
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = (22, 27, 34) # Dark CCTV theme
        cv2.putText(img, f"CCTV: {camera_title}", (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (120, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Live stream connecting...", (60, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 2, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes()

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
        force_fresh: bool = False
    ) -> CaptureResult:
        """
        Captures a live frame using the specified camera protocol.
        - Serializes requests per IP to protect camera RTSP socket limits.
        - Caches successful frames for 4.0 seconds for smooth UI polling.
        - Provides robust last-known-good frame fallback (up to 300s) to completely prevent black screens/flickering.
        """
        cache_key = self._get_cache_key(camera_type, stream_url, ip_address, port)
        target_ip = (ip_address or stream_url).split("://")[-1].split("@")[-1].split("/")[0].split(":")[0]
        now_ts = time.time()

        if not force_fresh:
            with self._cache_lock:
                if cache_key in self._snapshot_cache:
                    cached_time, cached_res = self._snapshot_cache[cache_key]
                    if (now_ts - cached_time) < 4.0 and cached_res.success and cached_res.image_bytes:
                        return cached_res

        # Acquire lock for this physical camera IP
        lock = self._get_ip_lock(target_ip)
        async with lock:
            # Re-check cache inside lock in case another coroutine just captured it
            if not force_fresh:
                with self._cache_lock:
                    if cache_key in self._snapshot_cache:
                        cached_time, cached_res = self._snapshot_cache[cache_key]
                        if (now_ts - cached_time) < 4.0 and cached_res.success and cached_res.image_bytes:
                            return cached_res

            adapter = self.adapters.get(camera_type, self.adapters[CameraTypeEnum.RTSP])

            logger.info(f"Capturing frame ({camera_type}) for Station {cs_id}/{cp_id} from {stream_url}")

            result = await adapter.capture(
                stream_url=stream_url,
                username=username,
                password=password,
                ip_address=ip_address,
                port=port
            )

            # If capture failed, attempt one fast retry (0.2s delay) to overcome packet loss
            if not result.success:
                await asyncio.sleep(0.2)
                retry_res = await adapter.capture(
                    stream_url=stream_url,
                    username=username,
                    password=password,
                    ip_address=ip_address,
                    port=port
                )
                if retry_res.success and retry_res.image_bytes:
                    result = retry_res

            if result.success and result.image_bytes:
                with self._cache_lock:
                    self._snapshot_cache[cache_key] = (now_ts, result)
                    self._last_known_good[cache_key] = (now_ts, result)
                return result

            # If live capture failed momentarily, serve last known good frame (up to 300s) to prevent black screen
            with self._cache_lock:
                if cache_key in self._last_known_good:
                    last_ts, last_res = self._last_known_good[cache_key]
                    if (now_ts - last_ts) < 300.0 and last_res.image_bytes:
                        logger.debug(f"Serving graceful fallback frame for {target_ip} (age: {now_ts - last_ts:.1f}s)")
                        return last_res

        if not result.success and self.enable_mock_fallback:
            logger.warning(
                f"Live camera capture failed ({result.error_message}). Using test generator fallback."
            )
            mock_result = MockCameraGenerator.generate_mock_frame(
                cs_id=cs_id,
                cp_id=cp_id,
                plate_number=expected_plate_number
            )
            mock_result.error_message = f"Simulated Fallback (Live Camera Error: {result.error_message})"
            return mock_result

        return result

    async def get_live_stream(
        self,
        camera_type: CameraTypeEnum,
        stream_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ip_address: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """Yields continuous MJPEG multipart stream for live browser preview."""
        adapter = self.adapters.get(camera_type, self.adapters[CameraTypeEnum.RTSP])
        if hasattr(adapter, "stream_frames"):
            async for chunk in adapter.stream_frames(
                stream_url=stream_url,
                username=username,
                password=password,
                ip_address=ip_address,
                port=port
            ):
                yield chunk
        else:
            delay = 0.2
            while True:
                res = await adapter.capture(
                    stream_url=stream_url,
                    username=username,
                    password=password,
                    ip_address=ip_address,
                    port=port
                )
                if res.success and res.image_bytes:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + res.image_bytes + b"\r\n")
                await asyncio.sleep(delay)


# Singleton instance (mock fallback disabled by default for production reliability)
camera_service = CameraService(enable_mock_fallback=False)
