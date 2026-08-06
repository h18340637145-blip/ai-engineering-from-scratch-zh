import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import CryptoAdvisor, MediaPipeline, TransportAdvisor, jni_is_justified
except ImportError:
    CryptoAdvisor = MediaPipeline = TransportAdvisor = jni_is_justified = None


class TestMediaJniNetworkAndSecurity(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(TransportAdvisor, "尚未实现多媒体、JNI 与网络模型")

    def test_reliable_stream_uses_tcp(self):
        self._require()
        self.assertEqual(TransportAdvisor.choose("stream", True), "TCP")

    def test_low_latency_unreliable_path_uses_udp(self):
        self._require()
        self.assertEqual(TransportAdvisor.choose("low-latency", False), "UDP")

    def test_bulk_data_uses_aes(self):
        self._require()
        self.assertEqual(CryptoAdvisor.recommend("bulk-data"), "AES")

    def test_signing_or_key_exchange_uses_asymmetric_crypto(self):
        self._require()
        self.assertEqual(CryptoAdvisor.recommend("signing"), "RSA/ECC")

    def test_media_pipeline_keeps_timestamp_stage(self):
        self._require()
        self.assertIn("timestamp", MediaPipeline.steps())

    def test_jni_requires_native_reuse_or_performance_need(self):
        self._require()
        self.assertFalse(jni_is_justified(False, False))
        self.assertTrue(jni_is_justified(True, False))

