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


class ParticipantPackRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = load_documents()
        self.original_texts = {
            document.filename: document.text for document in self.documents
        }
        self.load_documents_patch = patch(
            "retrieval.load_documents",
            return_value=self.documents,
        )
        self.load_documents_patch.start()
        self.addCleanup(self.load_documents_patch.stop)

    def test_participant_pack_question_benchmark(self) -> None:
        benchmark = (
            (
                "كم يستغرق تجديد دفتر العائلة؟",
                {"family_book_renewal.txt"},
                "الزمن المستغرق",
            ),
            (
                "وين بقدر أطلع جواز سفر بدل فاقد؟",
                {"passport_lost.txt"},
                "أماكن تقديم الخدمة",
            ),
            (
                "شو الأوراق المطلوبة لتجديد دفتر العائلة؟",
                {"family_book_renewal.txt"},
                "الوثائق المطلوبة",
            ),
            (
                "قديش صورة شخصية بحتاج للبطاقة الذكية لأول مرة، "
                "وقديش لتجديد جواز السفر؟",
                {"id_card_first_issue.txt", "passport_renewal.txt"},
                "الوثائق المطلوبة",
            ),
        )
        passed = 0
        for question, expected_sources, expected_section in benchmark:
            with self.subTest(question=question):
                results = retrieve(question, top_k=3)
                actual_sources = {result["source"] for result in results}
                self.assertTrue(expected_sources.issubset(actual_sources))
                self.assertTrue(
                    all(expected_section in result["text"] for result in results)
                )
                for result in results:
                    self.assertIn(
                        result["text"],
                        self.original_texts[result["source"]],
                    )
                passed += 1
        self.assertEqual(passed, len(benchmark))

    def test_confusing_first_issue_and_renewal_questions_stay_service_aware(
        self,
    ) -> None:
        cases = (
            (
                "شو الأوراق المطلوبة لدفتر العائلة لأول مرة؟",
                "family_book_first_issue.txt",
            ),
            (
                "شو الأوراق المطلوبة لتجديد دفتر العائلة؟",
                "family_book_renewal.txt",
            ),
            (
                "ما شروط جواز السفر لأول مرة؟",
                "passport_first_issue.txt",
            ),
            (
                "ما شروط تجديد جواز السفر؟",
                "passport_renewal.txt",
            ),
        )
        for question, expected_source in cases:
            with self.subTest(question=question):
                results = retrieve(question)
                self.assertTrue(results)
                self.assertEqual(results[0]["source"], expected_source)

    def test_lost_and_damaged_replacement_queries_do_not_mix(self) -> None:
        cases = (
            ("قديش رسوم جواز سفر ضايع؟", "passport_lost.txt"),
            ("شو رسوم جواز السفر الخربان؟", "passport_damaged.txt"),
            ("كم رسوم البطاقة الذكية المفقودة؟", "id_card_lost.txt"),
            ("كم رسوم البطاقة الذكية التالفة؟", "id_card_damaged.txt"),
        )
        for question, expected_source in cases:
            with self.subTest(question=question):
                results = retrieve(question)
                self.assertTrue(results)
                self.assertEqual(results[0]["source"], expected_source)
                self.assertTrue(
                    all("الرسوم" in result["text"] for result in results)
                )

    def test_issuance_word_does_not_override_lost_replacement_intent(self) -> None:
        results = retrieve("ما شروط إصدار جواز سفر بدل فاقد؟")

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "passport_lost.txt")
        self.assertTrue(
            all("شروط تقديم الخدمة" in result["text"] for result in results)
        )

    def test_exception_query_selects_notes_for_the_requested_service(self) -> None:
        results = retrieve("ما التعهد عند فقدان البطاقة الذكية؟")

        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "id_card_lost.txt")
        self.assertTrue(all("ملاحظات" in result["text"] for result in results))

    def test_multi_service_comparison_returns_evidence_for_each_service(self) -> None:
        question = (
            "قديش صورة شخصية بحتاج للبطاقة الذكية لأول مرة، "
            "وقديش لتجديد جواز السفر؟"
        )
        results = retrieve(question, top_k=2)

        self.assertEqual(
            {result["source"] for result in results},
            {"id_card_first_issue.txt", "passport_renewal.txt"},
        )
        for result in results:
            self.assertIn("الوثائق المطلوبة", result["text"])
            self.assertIn(result["text"], self.original_texts[result["source"]])

    def test_comparison_without_punctuation_returns_both_services(self) -> None:
        results = retrieve("قارن رسوم جواز السفر والبطاقة الذكية", top_k=2)

        self.assertEqual(
            {result["source"].rsplit("_", 1)[0] for result in results},
            {"passport", "id_card"},
        )
        self.assertTrue(all("الرسوم" in result["text"] for result in results))

    def test_unsupported_official_benchmark_question_returns_no_evidence(self) -> None:
        self.assertEqual(
            retrieve("كم سعر تذكرة الطيران من عمان إلى دبي؟"),
            [],
        )

    def test_service_specific_follow_up_without_history_returns_no_mixed_evidence(
        self,
    ) -> None:
        self.assertEqual(
            retrieve("وبقدر أقدّم الطلب من السفارة إذا كنت برا البلد؟"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
