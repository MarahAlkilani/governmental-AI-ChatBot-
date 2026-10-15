import unittest

from arabic_text import normalize_arabic, tokenize_arabic, tokenize_for_retrieval


class ArabicTextNormalizationTests(unittest.TestCase):
    def test_normalizes_alef_variants_diacritics_and_tatweel(self) -> None:
        self.assertEqual(
            normalize_arabic("إِصْدَارُ آدَمِيٍّ وأَلْف"),
            "اصدار ادمي والف",
        )
        self.assertEqual(
            normalize_arabic("ٱلْجَوَازُ"),
            "الجواز",
        )

    def test_normalizes_common_service_term_spelling_variations(self) -> None:
        first_spelling = tokenize_arabic("البِطَاقَة الذَّكِيَّة وإجراءات إصدار الجواز")
        alternate_spelling = tokenize_arabic("البطاقه الذكيه واجراءات اصدار الجواز")

        self.assertEqual(first_spelling, alternate_spelling)
        self.assertEqual(
            tokenize_arabic("دفتر العائلة بدل فاقد"),
            ["دفتر", "العايله", "بدل", "فاقد"],
        )

    def test_normalizes_arabic_and_persian_digits_to_western_digits(self) -> None:
        self.assertEqual(normalize_arabic("الرسوم: ١٢٣ دينار"), "الرسوم 123 دينار")
        self.assertEqual(normalize_arabic("الرسوم: ۱۲۳ دينار"), "الرسوم 123 دينار")
        self.assertEqual(
            tokenize_arabic("يستغرق ٩٠ يوماً؛ ويستغرق 90 يوما"),
            ["يستغرق", "90", "يوما", "ويستغرق", "90", "يوما"],
        )

    def test_normalizes_punctuation_and_whitespace(self) -> None:
        self.assertEqual(
            normalize_arabic("  جواز،   سفر!\n(بدل—فاقد)  "),
            "جواز سفر بدل فاقد",
        )
        self.assertEqual(
            tokenize_arabic("passport_lost.txt: جواز سفر"),
            ["passport", "lost", "txt", "جواز", "سفر"],
        )

    def test_normalization_does_not_mutate_original_evidence_text(self) -> None:
        original = "الرسوم: ١٠ دنانير، تُدفع عند تقديم الطلب."

        normalized = normalize_arabic(original)

        self.assertEqual(normalized, "الرسوم 10 دنانير تدفع عند تقديم الطلب")
        self.assertEqual(original, "الرسوم: ١٠ دنانير، تُدفع عند تقديم الطلب.")

    def test_retrieval_tokens_fold_common_colloquial_service_terms(self) -> None:
        self.assertEqual(
            tokenize_for_retrieval("وين بقدر أطلع باسبور ضايع؟"),
            ["مكان", "استطيع", "اطلع", "جواز", "فقدان"],
        )
        self.assertEqual(
            tokenize_for_retrieval("شو الأوراق المطلوبة لتجديد الهوية؟"),
            ["ماذا", "وثايق", "مطلوبه", "تجديد", "بطاقه"],
        )


if __name__ == "__main__":
    unittest.main()
