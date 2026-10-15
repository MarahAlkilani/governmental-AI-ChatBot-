import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingest import load_documents
from retrieval import retrieve


class BM25RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        data_dir = Path(self.temporary_directory.name)
        documents = {
            "passport_lost.txt": (
                "جواز سفر بدل فاقد\n"
                "شروط تقديم الخدمة\tيجب تقديم بلاغ فقدان من المركز الأمني.\n"
                "الوثائق المطلوبة للحصول على الخدمة\tبطاقة شخصية وبلاغ فقدان.\n"
                "الرسوم\tرسوم إصدار جواز السفر بدل فاقد ١٢٥ دينار.\n"
            ),
            "id_card_lost.txt": (
                "بطاقة الأحوال المدنية الذكية بدل فاقد\n"
                "شروط تقديم الخدمة\tيقدم صاحب العلاقة طلب البطاقة المفقودة.\n"
                "الوثائق المطلوبة للحصول على الخدمة\tنموذج البطاقة الشخصية.\n"
                "الزمن المستغرق لتقديم الخدمة\t20 دقيقة.\n"
                "الرسوم\tرسوم إصدار البطاقة الذكية بدل فاقد ٥ دنانير.\n"
            ),
            "family_book_renewal.txt": (
                "تجديد دفتر العائلة\n"
                "شروط تقديم الخدمة\tتجديد دفتر العائلة منتهي المدة.\n"
                "الوثائق المطلوبة للحصول على الخدمة\tدفتر العائلة المراد تجديده.\n"
                "الرسوم\tرسوم تجديد دفتر العائلة ديناران.\n"
            ),
            "passport_renewal.txt": (
                "تجديد جواز السفر\n"
                "شروط تقديم الخدمة\tتقديم طلب تجديد جواز السفر.\n"
                "الوثائق المطلوبة للحصول على الخدمة\tجواز السفر المراد تجديده.\n"
            ),
            "passport_damaged.txt": (
                "جواز سفر بدل تالف\n"
                "شروط تقديم الخدمة\tاحضار حجة وصاية لغايات تجديد جواز سفر لمن يلزمهم ذلك.\n"
                "الوثائق المطلوبة للحصول على الخدمة\tالبطاقة الشخصية لمتلقي الخدمة.\n"
            ),
        }
        for filename, text in documents.items():
            (data_dir / filename).write_bytes(text.encode("utf-8"))
        self.documents = load_documents(data_dir)
        self.original_texts = {
            document.filename: document.text for document in self.documents
        }
        self.load_documents_patch = patch(
            "retrieval.load_documents",
            return_value=self.documents,
        )
        self.load_documents_patch.start()
        self.addCleanup(self.load_documents_patch.stop)
        self.addCleanup(self.temporary_directory.cleanup)

    def test_retrieves_the_relevant_passage_for_arabic_queries(self) -> None:
        results = retrieve("كم رسوم إصدار جواز السفر بدل فاقد؟")

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "passport_lost.txt")
        self.assertIn("١٢٥ دينار", results[0]["text"])
        self.assertIsInstance(results[0]["score"], float)
        self.assertTrue(all("الرسوم" in result["text"] for result in results))

    def test_retrieves_duration_sections_for_duration_questions(self) -> None:
        results = retrieve("كم يستغرق إصدار البطاقة الذكية؟")

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "id_card_lost.txt")
        self.assertIn("الزمن المستغرق", results[0]["text"])
        self.assertTrue(
            all("الزمن المستغرق" in result["text"] for result in results)
        )

    def test_service_identity_ranks_renewal_ahead_of_incidental_mentions(self) -> None:
        results = retrieve("ما شروط تجديد جواز السفر؟")

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "passport_renewal.txt")

    def test_retrieves_requested_documents_and_preserves_exact_excerpts(self) -> None:
        results = retrieve("ما الوثائق المطلوبة لبطاقة ذكية بدل فاقد؟", top_k=3)

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "id_card_lost.txt")
        self.assertIn("الوثائق المطلوبة", results[0]["text"])
        self.assertTrue(
            all("الوثائق المطلوبة" in result["text"] for result in results)
        )
        for result in results:
            self.assertIn(result["source"], self.original_texts)
            self.assertIn(
                result["text"],
                self.original_texts[result["source"]],
            )

    def test_does_not_fill_top_k_with_unrelated_passages(self) -> None:
        self.assertEqual(retrieve("كيف أزرع الطماطم في الحديقة؟", top_k=10), [])

    def test_empty_question_and_invalid_top_k_return_no_results(self) -> None:
        self.assertEqual(retrieve("   "), [])
        for invalid_top_k in (0, -1, 1.5, "3", True, None):
            with self.subTest(top_k=invalid_top_k):
                self.assertEqual(
                    retrieve("رسوم جواز السفر بدل فاقد", top_k=invalid_top_k),
                    [],
                )

    def test_empty_documents_and_sections_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "empty.txt").write_bytes(b"")
            (data_dir / "empty_sections.txt").write_bytes(
                "جواز سفر\nالرسوم\nشروط تقديم الخدمة\n".encode("utf-8")
            )
            empty_documents = load_documents(data_dir)

            with patch("retrieval.load_documents", return_value=empty_documents):
                self.assertEqual(retrieve("رسوم جواز السفر"), [])

    def test_results_have_only_the_public_fields_and_are_deterministic(self) -> None:
        first = retrieve("رسوم دفتر العائلة تجديد")
        second = retrieve("رسوم دفتر العائلة تجديد")

        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(
            all(set(result) == {"text", "source", "score"} for result in first)
        )


if __name__ == "__main__":
    unittest.main()
