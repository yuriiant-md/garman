import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.documents.masking import mask_pii


def masked(text: str, extra_names: tuple[str, ...] = ()) -> str:
    return mask_pii(text, extra_names)[0]


def test_masks_iban_including_ocr_damaged_country_code():
    text = "IBAN UA303808050000000026000571797\nр/р ПА068201720343140002000003097"

    result = masked(text)

    assert result == "IBAN [IBAN]\nр/р [IBAN]"


def test_masks_iban_written_with_spaces():
    assert masked("UA30 3808 0500 0000 0026 0005 7179 7.") == "[IBAN]."


def test_masks_iban_whose_country_code_was_lost():
    assert masked("р/р 303808050000000026000571797,") == "р/р [IBAN],"


def test_masks_iban_whose_country_code_ocr_read_as_three_letters():
    text = "р/р ІІА658201720344210021000095135\nIIА51 3005 8400 0002 6007 2002 4500 8"

    assert masked(text) == "р/р І[IBAN]\nI[IBAN]"


def test_masks_old_bank_account_number():
    assert masked("р/р 26004123456789 в банку") == "р/р [РАХУНОК] в банку"


def test_masks_short_old_account_after_account_prefix():
    assert masked("Р/р 26009477104 в банку") == "Р/р [РАХУНОК] в банку"


def test_masks_twelve_digit_vat_number():
    assert masked("ІПН 046873010150\n") == "ІПН [ІПН]\n"


@pytest.mark.parametrize(
    "phone",
    ["+380966378279", "+38 (096) 637-82-79", "(044) 123-45-67", "096 637 82 79", "0966378279"],
)
def test_masks_phone_formats(phone):
    assert masked(f"Тел. {phone}, факс") == "Тел. [ТЕЛЕФОН], факс"


def test_masks_ten_digit_tax_number_but_keeps_eight_digit_company_code():
    text = "Код ЄДРПОУ 3063117114\nкод ЄДРПОУ 08562915"

    assert masked(text) == "Код ЄДРПОУ [РНОКПП]\nкод ЄДРПОУ 08562915"


def test_masks_email():
    assert masked("e-mail: zerno.fop@ukr.net;") == "e-mail: [EMAIL];"


def test_masks_full_name_and_every_later_form_of_the_surname():
    text = (
        "Фізична особа підприємець Петренко Іван Степанович (далі - Постачальник)\n"
        "ПЕТРЕНКО\n"
        "І.С. Петренко\n"
        "ФОП «Петренко Іван Степанович»"
    )

    result = masked(text)

    assert "Петренко" not in result and "ПЕТРЕНКО" not in result
    assert "Іван" not in result and "Степанович" not in result
    assert result.startswith("Фізична особа підприємець [ПІБ] (далі - Постачальник)")


def test_masks_declined_name_after_job_title():
    text = "в особі начальника Коваленка Віталія Володимировича, що діє\nВ.В. Коваленко"

    result = masked(text)

    assert result == "в особі начальника [ПІБ], що діє\n[ПІБ]"


def test_masks_name_written_first_name_first():
    text = "Директор Олена Петрівна Мирошник\nМирошник О.П."

    result = masked(text)

    assert result == "Директор [ПІБ]\n[ПІБ]"


def test_masks_first_name_and_surname_without_patronymic():
    text = "в особі Директор Степан Гордієнко, що діє\nСтепан Гордієнко"

    assert masked(text) == "в особі Директор [ПІБ], що діє\n[ПІБ]"


def test_masks_lone_patronymic_and_lone_first_name_in_signature_noise():
    text = "[ПІБ] IOpia Олександровича. що діє\nСЕРГІЙ\n! ВОЛОДИМИРОВИЧ1"

    result = masked(text)

    assert result == "[ПІБ] IOpia [ПІБ]. що діє\n[ПІБ]\n! [ПІБ]1"


def test_keeps_place_names_and_words_that_look_like_names():
    text = "с. Романово; Рівні частини; дорівнює 5%; Працівники складу"

    assert masked(text) == text


def test_keeps_job_title_before_name():
    assert masked("Начальник Сидоренко В. В.").startswith("Начальник [ПІБ]")


def test_keeps_person_name_inside_institution_name():
    text = "Дослідна станція помології імені Л.П.Симиренка; Заповідник ім. Ф.Е. Фальц-Фейна"

    assert masked(text) == text


def test_keeps_institution_name_when_ocr_garbled_imeni():
    text = "дослідна станція iMCHi М. I. Вавилова Інституту\nстанція iivieni М.І. Вавилова"

    assert masked(text) == text


def test_keeps_street_named_after_person():
    text = "м. Київ, вул. Петра Сагайдачного 1; вулиця Тараса Шевченка, 5"

    assert masked(text) == text


def test_masks_name_after_initials_of_the_previous_name():
    # initials "І.В." before the next name are not the word "імені"
    text = "у складі суддів: Булгакової І.В., Васьковського О.В., Чумака Ю.Я."

    assert masked(text) == "у складі суддів: [ПІБ], [ПІБ], [ПІБ]"


def test_still_masks_name_after_conjunction_i():
    assert masked("підписали Петренко І.С. і В.В. Коваленко") == "підписали [ПІБ] і [ПІБ]"


def test_keeps_company_names_and_amounts():
    text = "ТОВ «Агро Світ», ФГ «Садок вишневий»: 60 000,00 грн за 10000 кг, 01.06.2010 р."

    assert masked(text) == text


def test_masks_extra_names_given_by_reviewer():
    assert masked("підпис Гнатюка та Гнатюк", extra_names=("Гнатюк",)) == "підпис [ПІБ] та [ПІБ]"


def test_reports_masked_values_in_the_same_order_under_any_hash_seed():
    # surnames of one length: a set of their stems iterates in an order the hash seed picks
    script = (
        "from src.documents.masking import mask_pii\n"
        "names = ('Чумак', 'Бойко', 'Дячук', 'Кузик', 'Гнатюк', 'Пєсков')\n"
        "print(mask_pii('Підписали ' + ', '.join(names) + '.', names)[1])\n"
    )
    root = Path(__file__).resolve().parents[1]

    orders = {
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in range(8)
    }

    assert len(orders) == 1


def test_reports_each_masked_value_with_its_kind():
    _, found = mask_pii("Тел. 0966378279, Петренко Іван Степанович")

    assert ("ТЕЛЕФОН", "0966378279") in found
    assert ("ПІБ", "Петренко Іван Степанович") in found


def test_company_codes_are_hidden_only_when_asked():
    text = "код ЄДРПОУ 30307207 ІПН\nвул. Миру, буд. 40, 12345678 (надалі – місце поставки)\n"

    hidden, found = mask_pii(text, company_codes=True)

    assert hidden == (
        "код ЄДРПОУ [ЄДРПОУ] ІПН\nвул. Миру, буд. 40, [ЄДРПОУ] (надалі – місце поставки)\n"
    )
    assert [kind for kind, _ in found] == ["ЄДРПОУ", "ЄДРПОУ"]
    assert mask_pii(text)[0] == text  # Prozorro buyers are public bodies: their codes stay


def test_company_codes_spare_amounts_and_contract_numbers():
    text = "Сума 12500000,00 грн, аванс 12500000.00 грн, договір № 20260484\n"

    assert mask_pii(text, company_codes=True)[0] == text


def test_reviewer_strings_are_hidden_even_when_split_across_lines():
    hidden, found = mask_pii("Agro-zbud\nTOV ДОГОВІР ПОСТАВКИ\n", extra_hidden=("Agro-zbud TOV",))

    assert hidden == "[ПРИХОВАНО] ДОГОВІР ПОСТАВКИ\n"
    assert found == [("ПРИХОВАНО", "Agro-zbud\nTOV")]


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        (
            'Директор Іван Степанович ТОВ "Агро" поставляє товар. Товариство, товарів 100 т.',
            ["ТОВ", "товар.", "Товариство", "товарів"],
        ),
        (
            "Василя Петровича Договір укладено. Договір № 5, договірна ціна",
            ["Договір укладено", "Договір № 5", "договірна"],
        ),
    ],
)
def test_word_after_name_that_is_a_common_word_is_not_taken_as_surname(text, kept):
    result = masked(text)

    assert all(word in result for word in kept)
    assert result.count("[ПІБ]") == 1


@pytest.mark.parametrize(
    "phone",
    [
        "тел. (04137) 5-4.3-08",
        "Тел (Ьакс {0416119-13-5 8",
        "Тел/факс (04161)9-13-58",
        "тел./факс (04345) 2-31-98, 2-16-98",
        "тел. (0512) 34-56-78",
        "050.123.45.67",
        "050/123-45-67",
    ],
)
def test_masks_landline_and_dotted_phone_formats(phone):
    result = masked(f"{phone}\nкод ЄДРПОУ 08562915")

    assert not any(char.isdigit() for char in result.splitlines()[0])
    assert result.endswith("код ЄДРПОУ 08562915")


def test_phone_cue_inside_other_words_keeps_quantities_and_dates():
    text = "автомобіля 10 000 кг; мобілізації з 24.02.2022; факсимільного 01.01.2024 р."

    assert masked(text) == text


def test_masks_surname_forms_with_dropped_vowel():
    text = "Кравець О.І.\nпідписав Кравцю, разом з Кравцем"

    assert masked(text) == "[ПІБ]\nпідписав [ПІБ], разом з [ПІБ]"


def test_keeps_ten_digit_contract_number_after_number_sign():
    assert masked("Договір № 2024100012 від 12.03.2024") == "Договір № 2024100012 від 12.03.2024"


def test_masks_extra_address_even_when_split_across_lines():
    text = "ФОП [ПІБ]\nсмт. Іванівка, вул.\nСадова 7А, тел"

    result, found = mask_pii(text, extra_addresses=("смт. Іванівка, вул. Садова 7А",))

    assert result == "ФОП [ПІБ]\n[АДРЕСА], тел"
    assert ("АДРЕСА", "смт. Іванівка, вул.\nСадова 7А") in found
