import logging
import tempfile
import unittest
from pathlib import Path

from ingest import load_documents


class LoadDocumentsTests(unittest.TestCase):
    def test_discovers_nested_documents_and_extracts_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            document_path = data_dir / "services" / "passport_lost.txt"
            document_path.parent.mkdir()
            document_path.write_bytes(
                "الرئيسية\nدليل الخدمات\nجواز سفر بدل فاقد\n"
                "الفئة المستفيدة\tالمواطنون\nالرسوم\n5 دنانير\n".encode("utf-8")
            )

            documents = load_documents(data_dir)

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(document.filename, "passport_lost.txt")
        self.assertEqual(document.service_id, "passport_lost")
        self.assertEqual(document.service_name, "جواز سفر بدل فاقد")
        self.assertEqual(
            [section.title for section in document.sections],
            ["الفئة المستفيدة", "الرسوم"],
        )
        self.assertEqual(
            document.sections[0].text,
            "الفئة المستفيدة\tالمواطنون\n",
        )
        self.assertEqual(document.sections[1].text, "الرسوم\n5 دنانير\n")

    def test_preserves_original_text_including_line_endings_and_whitespace(self) -> None:
        original_text = (
            "جواز سفر بدل فاقد\r\n"
            "الوثائق المطلوبة للحصول على الخدمة\tصورة عن الهوية  \r\n"
            "\r\nالنص الأخير بلا سطر جديد"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "service.txt"
            path.write_bytes(original_text.encode("utf-8"))

            document = load_documents(temporary_directory)[0]

        self.assertEqual(document.text, original_text)

    def test_warns_and_skips_invalid_utf8_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_document = Path(temporary_directory) / "invalid.txt"
            invalid_document.write_bytes(b"\xff\xfe")

            with self.assertLogs("ingest", level=logging.WARNING):
                documents = load_documents(temporary_directory)

        self.assertEqual(documents, [])

    def test_missing_directory_returns_empty_list_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_directory = Path(temporary_directory) / "missing"

            with self.assertLogs("ingest", level=logging.WARNING):
                documents = load_documents(missing_directory)

        self.assertEqual(documents, [])


if __name__ == "__main__":
    unittest.main()
