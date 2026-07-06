# main.py

import time

from camera import Camera
from ocr import OCRProcessor
from corrector import TextCorrector
from tts import TextToSpeech


# ============================================================
# Configuration
# ============================================================

USE_CORRECTION = True

OCR_INTERVAL = 0.5   # seconds between live OCR updates


PIPER_MODEL = (
    "./model_piper/en_US-amy-medium.onnx"
)

OUTPUT_AUDIO = (
    "output/audio.wav"
)


# ============================================================
# Main
# ============================================================

def main():

    global USE_CORRECTION


    print("=" * 70)
    print("Camera OCR → Corrector → Piper")
    print("=" * 70)


    # --------------------------------------------------------
    # Load modules
    # --------------------------------------------------------

    camera = Camera(
        camera_index=0,
        width=1280,
        height=720,
        show_preview=True,
    )


    ocr = OCRProcessor(
        language="en",
        confidence_threshold=0.7,
        debug=True,
    )


    corrector = TextCorrector(
        debug=True,
    )


    tts = TextToSpeech(
        model_path=PIPER_MODEL,
        output_path=OUTPUT_AUDIO,
        auto_play=True,
    )


    print("\nReady.")
    print("SPACE/ENTER : Read text")
    print("C           : Toggle correction")
    print("Q           : Quit")


    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    last_ocr_time = 0

    latest_frame = None
    latest_text = ""
    latest_detections = []


    try:

        while True:

            # -----------------------------------------------
            # Camera
            # -----------------------------------------------

            frame, timestamp = camera.get_frame()


            if frame is None:
                continue


            latest_frame = frame


            # -----------------------------------------------
            # Live OCR detection
            # -----------------------------------------------

            now = time.time()


            if (
                now - last_ocr_time
                >= OCR_INTERVAL
            ):

                last_ocr_time = now


                result = (
                    ocr.extract_text(
                        frame,
                        return_detections=True
                    )
                )


                if isinstance(result, tuple):

                    latest_text = result[0]
                    latest_detections = result[1]

                else:

                    # fallback compatibility
                    latest_text = result
                    latest_detections = []


            # -----------------------------------------------
            # Preview
            # -----------------------------------------------

            camera.preview(
                frame,
                latest_detections,
            )


            key = camera.read_key()


            # -----------------------------------------------
            # Quit
            # -----------------------------------------------

            if key == "quit":
                break


            # -----------------------------------------------
            # Toggle correction
            # -----------------------------------------------

            if key == "toggle":

                USE_CORRECTION = (
                    not USE_CORRECTION
                )

                print(
                    "\nCorrection:",
                    "ON"
                    if USE_CORRECTION
                    else "OFF"
                )


            # -----------------------------------------------
            # Process selected frame
            # -----------------------------------------------

            if key == "capture":

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    "RAW OCR"
                )

                print(
                    "=" * 70
                )


                text = latest_text


                print(text)


                if not text.strip():

                    print(
                        "No text detected."
                    )

                    continue


                # -------------------------------
                # Correction
                # -------------------------------

                if USE_CORRECTION:


                    print(
                        "\n"
                        + "=" * 70
                    )

                    print(
                        "CORRECTION"
                    )

                    print(
                        "=" * 70
                    )


                    start = time.perf_counter()


                    final_text = (
                        corrector.correct(
                            text
                        )
                    )


                    elapsed = (
                        time.perf_counter()
                        -
                        start
                    )


                    print(
                        "\nCorrected:"
                    )

                    print(
                        final_text
                    )


                    print(
                        f"\nCorrection time: "
                        f"{elapsed:.3f}s"
                    )


                else:

                    print(
                        "\nCorrection disabled."
                    )

                    final_text = text


                # -------------------------------
                # TTS
                # -------------------------------

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    "TTS"
                )

                print(
                    "=" * 70
                )


                audio = (
                    tts.synthesize(
                        final_text
                    )
                )


                print(
                    f"Saved: {audio}"
                )


    finally:

        camera.release()


# ============================================================

if __name__ == "__main__":
    main()