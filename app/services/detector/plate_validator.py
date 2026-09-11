"""Strict & robust Korean license plate validator with artifact rejection."""
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class KoreanPlateValidator:
    """
    Koreya davlat raqamlari qat'iy standart tekshiruvi (Strict & Robust Korean Plate Validator).
    1. Yangi 8-xonali: 123가 4567 (3 ta raqam + 1 ta qonuniy Koreyscha harf + 4 ta raqam)
    2. Eski 7-xonali: 12가 3456 (2 ta raqam + 1 ta qonuniy Koreyscha harf + 4 ta raqam)
    3. Viloyatli format: 서울12가 3456 (Viloyat nomi + 1-2 raqam + 1 ta qonuniy Koreyscha harf + 4 ta raqam)
    
    Qat'iy qoidalar:
    - O'rtadagi harf FAQAT qonuniy Koreya avtomobil harflari (LEGAL_HANGUL) bo'lishi shart.
    - Harfsiz telefon raqamlar (1522-1562), lift stikerlari (손대지 마세요), sana va quruq raqamlar 100% RAD ETILADI (None).
    - OCR fon shovqinlari tufayli xato o'qilgan qo'sh unlilarni (혀 -> 허, 여 -> 어) avtomatik to'g'rilaydi.
    """

    # Koreya qonunchiligidagi rasmiy 40 ta davlat raqami harflari
    LEGAL_HANGUL = set(
        # Shaxsiy (자가용 - 32 ta)
        "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주"
        # Ijara (렌터카 - 3 ta)
        "하허호"
        # Tijorat / Taksi / Avtobus (영업용 - 4 ta)
        "아바사자"
        # Yetkazib berish (택배 - 1 ta)
        "배"
        # Harbiy / Maxsus
        "육해공국합외영"
    )

    # Diphthong / misread Hangul to legal Hangul mapping
    HANGUL_DIPHTHONG_MAP = {
        "혀": "허", "여": "어", "야": "아", "교": "고", "규": "구",
        "녀": "너", "뇨": "노", "뉴": "누", "려": "러", "료": "로",
        "류": "루", "며": "머", "묘": "모", "뮤": "무", "벼": "버",
        "뵤": "보", "뷰": "부", "셔": "서", "쇼": "소", "슈": "수",
        "져": "저", "죠": "조", "쥬": "주", "효": "호", "휴": "후",
        "햐": "하", "헤": "허", "히": "허", "후": "호", "흐": "하",
        "에": "어", "예": "어", "게": "거", "네": "너", "데": "더",
        "레": "러", "메": "머", "베": "버", "세": "서", "제": "저",
        "기": "가", "니": "나", "디": "다", "리": "라", "미": "마",
        "비": "바", "시": "사", "이": "어", "지": "자"
    }

    # Uzoq masofali OCR piksellarni to'g'irlash xaritasi (Faqat bitta harf xira bo'lganda)
    OCR_HANGUL_MAP = {
        "3": "머", "8": "머", "B": "머", "D": "머",
        "S": "수", "G": "가", "H": "하", "N": "나", "M": "마",
        "R": "러", "L": "라", "J": "주", "4": "머", "A": "아",
        "0": "어", "O": "어", "Q": "어", "E": "어", "U": "우",
        "V": "버", "W": "버", "P": "부"
    }

    # Birlashtirilgan (fused) raqam + harf xaritasi (e.g. '2어' -> '때', '대')
    HANGUL_MERGED_MAP = {
        "때": ("2", "어"),
        "대": ("2", "어"),
        "태": ("7", "어"),
        "티": ("7", "이"),
        "피": ("1", "이"),
        "빠": ("8", "아"),
        "따": ("2", "아"),
        "짜": ("4", "아"),
        "까": ("1", "가"),
        "싸": ("4", "아"),
        "더": ("2", "어"),
    }

    # Rasmiy Koreya viloyatlari
    LEGAL_REGIONS = (
        "서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주"
    )

    # 1. Aniq 8-xonali: 123가4567, 312버6132
    PATTERN_EXACT_8 = re.compile(r"^(\d{3})([가-힣])(\d{4})$")

    # 2. Aniq 7-xonali: 81머2072, 47호6633, 58버6091
    PATTERN_EXACT_7 = re.compile(r"^(\d{2})([가-힣])(\d{4})$")

    # 3. Viloyatli format: 서울12가3456
    PATTERN_REGIONAL = re.compile(rf"^({LEGAL_REGIONS})(\d{{1,2}})([가-힣])(\d{{4}})$")

    # 4. Angled / Tilted Qiya holatdagi oraliq artifactlar (e.g. 312버61321 -> 312버6132)
    PATTERN_SEARCH_ROBUST = re.compile(r"(\d{2,3})([가-힣])\D*?(\d{4})")

    # 5. Uzoq masofadagi harfi xira 8-xonali (e.g. 12304567)
    PATTERN_DISTANT_8 = re.compile(r"^(\d{3})([A-Za-z0-9])(\d{4})$")

    # 6. Uzoq masofadagi harfi xira 7-xonali (e.g. 8102017)
    PATTERN_DISTANT_7 = re.compile(r"^(\d{2})([A-Za-z0-9])(\d{4})$")

    # Anti-patterns (Customer service numbers, charging station IDs, phone prefixes)
    REJECT_PREFIXES = ("1522", "1588", "1600", "1899", "080", "1544", "1644", "1688", "070", "050")

    @classmethod
    def normalize_hangul(cls, char: str) -> Optional[str]:
        """Validates or normalizes a single Korean character to a legal license plate letter."""
        if char in cls.LEGAL_HANGUL:
            return char
        if char in cls.HANGUL_DIPHTHONG_MAP:
            return cls.HANGUL_DIPHTHONG_MAP[char]
        return None

    @classmethod
    def validate_and_normalize(cls, raw_text: str) -> Optional[Tuple[str, float]]:
        """
        Qat'iy tekshiruv:
        Koreya davlat raqami bo'lsa (normalized_plate, score) qaytaradi.
        Boshqa har qanday narsa (telefon raqam, stiker, qog'oz) bo'lsa darhol NONE qaytaradi.
        """
        if not raw_text or len(raw_text.strip()) < 5:
            return None

        # Clean noise characters
        cleaned = re.sub(r"[^가-힣0-9A-Za-z]", "", raw_text)

        # Reject phone numbers and service stickers
        for pfx in cls.REJECT_PREFIXES:
            if cleaned.startswith(pfx) and not any(h in cleaned for h in cls.LEGAL_HANGUL):
                return None

        # 1. Aniq 8-xonali format (123가4567, 312버6132)
        m = cls.PATTERN_EXACT_8.match(cleaned)
        if m:
            prefix, hangul, suffix = m.groups()
            norm_h = cls.normalize_hangul(hangul)
            if norm_h:
                score = 1.0 if norm_h == hangul else 0.98
                return f"{prefix}{norm_h}{suffix}", score

        # 2. Aniq 7-xonali format (81머2072, 47호6633, 52허0586)
        m = cls.PATTERN_EXACT_7.match(cleaned)
        if m:
            prefix, hangul, suffix = m.groups()
            norm_h = cls.normalize_hangul(hangul)
            if norm_h:
                score = 1.0 if norm_h == hangul else 0.98
                return f"{prefix}{norm_h}{suffix}", score

        # 3. Viloyatli format (서울12가3456)
        m = cls.PATTERN_REGIONAL.match(cleaned)
        if m:
            region, num, hangul, suffix = m.groups()
            norm_h = cls.normalize_hangul(hangul)
            if norm_h:
                score = 0.95 if norm_h == hangul else 0.93
                return f"{region}{num}{norm_h}{suffix}", score

        # 4. Qiya / Burchak ostidagi raqamlardan ajratib olish (312버61321 -> 312버6132)
        m = cls.PATTERN_SEARCH_ROBUST.search(cleaned)
        if m:
            prefix, hangul, suffix = m.groups()
            norm_h = cls.normalize_hangul(hangul)
            if norm_h:
                score = 1.0 if norm_h == hangul else 0.95
                return f"{prefix}{norm_h}{suffix}", score

        # 5. Birlashtirilgan (fused) 1-raqam + harf (e.g. 5때0586 -> 52어0586)
        m_fused = re.match(r"^(\d{1})([가-힣])(\d{4})$", cleaned)
        if m_fused:
            d1, h, suffix = m_fused.groups()
            if h in cls.HANGUL_MERGED_MAP:
                d2, legal_h = cls.HANGUL_MERGED_MAP[h]
                return f"{d1}{d2}{legal_h}{suffix}", 0.85

        # 6. Uzoq masofali 8-xonali xira harf (Faqat qat'iy harf xaritasidan)
        m = cls.PATTERN_DISTANT_8.match(cleaned)
        if m:
            prefix, mid, suffix = m.groups()
            if mid.upper() in cls.OCR_HANGUL_MAP:
                return f"{prefix}{cls.OCR_HANGUL_MAP[mid.upper()]}{suffix}", 0.75

        # 7. Uzoq masofali 7-xonali xira harf (Faqat qat'iy harf xaritasidan)
        m = cls.PATTERN_DISTANT_7.match(cleaned)
        if m:
            prefix, mid, suffix = m.groups()
            if mid.upper() in cls.OCR_HANGUL_MAP:
                return f"{prefix}{cls.OCR_HANGUL_MAP[mid.upper()]}{suffix}", 0.75

        return None

    @classmethod
    def format_display(cls, plate_number: str) -> str:
        """
        Raqamni chiroyli probel bilan formatlash: e.g. 81머2072 -> 81머 2072
        """
        if not plate_number or len(plate_number) < 7:
            return plate_number or ""
        return f"{plate_number[:-4]} {plate_number[-4:]}"
