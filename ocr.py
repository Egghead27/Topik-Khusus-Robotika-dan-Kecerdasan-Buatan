# ocr.py

from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(
        self,
        language="en",
        confidence_threshold=0.7,
        line_merge_threshold=20,
        debug=False,
    ):
        """
        OCR processor using PaddleOCR.

        Args:
            language:
                OCR language

            confidence_threshold:
                Ignore OCR results below this confidence

            line_merge_threshold:
                Y distance used for grouping text lines

            debug:
                Print OCR detections
        """

        self.confidence_threshold = confidence_threshold
        self.line_merge_threshold = line_merge_threshold
        self.debug = debug


        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=language,
        )


    # ========================================================
    # Public API
    # ========================================================

    def extract_text(
        self,
        frame,
        return_detections=False,
    ):
        """
        Extract text from OpenCV image.

        Args:
            frame:
                OpenCV BGR image

            return_detections:
                False:
                    return text only

                True:
                    return (
                        text,
                        detections
                    )

        Returns:

        text

        or

        (
            text,
            [
                {
                    text,
                    confidence,
                    box,
                    x,
                    y,
                    line
                }
            ]
        )
        """

        try:

            result = self.ocr.predict(
                frame
            )


            if not result:

                if return_detections:
                    return "", []

                return ""


            detections = []


            # ------------------------------------------------
            # Extract OCR results
            # ------------------------------------------------

            for page in result:

                rec_texts = page.get(
                    "rec_texts",
                    []
                )

                rec_scores = page.get(
                    "rec_scores",
                    []
                )

                rec_polys = page.get(
                    "rec_polys",
                    []
                )


                for text, score, poly in zip(
                    rec_texts,
                    rec_scores,
                    rec_polys,
                ):

                    if (
                        score
                        <
                        self.confidence_threshold
                    ):
                        continue


                    text = text.strip()


                    if not text:
                        continue


                    # top-left approximation

                    x = min(
                        point[0]
                        for point in poly
                    )

                    y = min(
                        point[1]
                        for point in poly
                    )


                    line = round(
                        y
                        /
                        self.line_merge_threshold
                    )


                    detections.append(
                        {
                            "text": text,
                            "confidence": float(score),

                            # for sorting
                            "x": x,
                            "y": y,
                            "line": line,

                            # for camera bbox
                            "box": poly,
                        }
                    )


                    if self.debug:

                        print(
                            "[OCR]",
                            f"'{text}'",
                            f"conf={score:.2f}",
                            f"line={line}",
                        )


            if not detections:

                if return_detections:
                    return "", []

                return ""


            # ------------------------------------------------
            # Reading order
            # ------------------------------------------------

            detections.sort(
                key=lambda item:
                (
                    item["line"],
                    item["x"],
                )
            )


            # ------------------------------------------------
            # Reconstruct text
            # ------------------------------------------------

            lines = {}


            for det in detections:

                line = det["line"]


                if line not in lines:
                    lines[line] = []


                lines[line].append(
                    det["text"]
                )


            reconstructed = []


            for line in sorted(lines):

                reconstructed.append(
                    " ".join(
                        lines[line]
                    )
                )


            text = "\n".join(
                reconstructed
            )


            text = text.strip()


            # ------------------------------------------------
            # Return
            # ------------------------------------------------

            if return_detections:

                return (
                    text,
                    detections
                )


            return text


        except Exception as e:

            if self.debug:

                print(
                    f"[OCR ERROR] {e}"
                )


            if return_detections:

                return "", []


            return ""