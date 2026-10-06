import unittest

from hiremate.audio import AudioIO


class AudioIOTests(unittest.IsolatedAsyncioTestCase):
    async def test_play_rejects_incomplete_pcm_sample(self) -> None:
        audio = AudioIO()

        with self.assertRaisesRegex(ValueError, "incomplete PCM16 sample"):
            await audio.play(b"\x01")

    async def test_play_queues_pcm_for_speaker_callback(self) -> None:
        audio = AudioIO()
        await audio.play(b"\x01\x02\x03\x04")
        output = bytearray(8)

        audio._on_speaker_audio(output, 4, None, None)

        self.assertEqual(output, b"\x01\x02\x03\x04\x00\x00\x00\x00")

    async def test_interrupt_discards_queued_and_partially_played_audio(self) -> None:
        audio = AudioIO()
        await audio.play(b"\x01\x02\x03\x04")
        first_output = bytearray(2)
        audio._on_speaker_audio(first_output, 1, None, None)
        self.assertEqual(first_output, b"\x01\x02")

        audio.interrupt_playback()

        remaining_output = bytearray(6)
        audio._on_speaker_audio(remaining_output, 3, None, None)
        self.assertEqual(remaining_output, bytes(6))

        await audio.play(b"\x05\x06")
        next_output = bytearray(2)
        audio._on_speaker_audio(next_output, 1, None, None)
        self.assertEqual(next_output, b"\x05\x06")

    async def test_microphone_frames_are_available_to_async_sender(self) -> None:
        audio = AudioIO()
        audio._enqueue_microphone_frame(b"\x01\x02")

        frame = await anext(audio.microphone_frames())

        self.assertEqual(frame, b"\x01\x02")


if __name__ == "__main__":
    unittest.main()
