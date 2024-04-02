import math
import datetime


def calculate_biorhythm(birth_date, today):
    physical = lambda x: round(100 * math.sin(2 * math.pi * x / 23), 2)
    emotional = lambda x: round(100 * math.sin(2 * math.pi * x / 28), 2)
    intellectual = lambda x: round(100 * math.sin(2 * math.pi * x / 33), 2)

    days_alive = (today - birth_date).days
    physical_score = physical(days_alive)
    emotional_score = emotional(days_alive)
    intellectual_score = intellectual(days_alive)

    return physical_score, emotional_score, intellectual_score


def print_biorhythm_results(physical, emotional, intellectual):
    print("Twoje wyniki biorytmów:")
    print("Fizyczne: ", physical)
    print("Emocjonalne: ", emotional)
    print("Intelektualne: ", intellectual)


def main():
    # Pobranie danych od użytkownika
    name = input("Jak się nazywasz? ")
    year = int(input("W którym roku się urodziłeś? "))
    month = int(input("W którym miesiącu się urodziłeś? "))
    day = int(input("W którym dniu miesiąca się urodziłeś? "))

    birth_date = datetime.date(year, month, day)
    today = datetime.date.today()

    # Obliczenie biorytmów
    physical, emotional, intellectual = calculate_biorhythm(birth_date, today)

    # Wyświetlenie wyników biorytmów
    print("\nWitaj,", name, "!")
    print(
        "Dzisiaj jest",
        today.strftime("%d/%m/%Y"),
        "i to jest",
        (today - birth_date).days,
        "dzień Twojego życia.",
    )
    print_biorhythm_results(physical, emotional, intellectual)

    # Sprawdzenie wyników biorytmów
    if physical > 0.5:
        print("\nGratulacje! Twój wynik biorytmu fizycznego jest wysoki.")
    elif physical < -0.5:
        print("\nOj, niski wynik biorytmu fizycznego. Trzymaj się!")
        next_physical = calculate_biorhythm(
            birth_date, today + datetime.timedelta(days=1)
        )[0]
        if next_physical > physical:
            print("Nie martw się. Jutro będzie lepiej!")
    if emotional > 0.5:
        print("Gratulacje! Twój wynik biorytmu emocjonalnego jest wysoki.")
    elif emotional < -0.5:
        print("Oj, niski wynik biorytmu emocjonalnego. Trzymaj się!")
        next_emotional = calculate_biorhythm(
            birth_date, today + datetime.timedelta(days=1)
        )[1]
        if next_emotional > emotional:
            print("Nie martw się. Jutro będzie lepiej!")
    if intellectual > 0.5:
        print("Gratulacje! Twój wynik biorytmu intelektualnego jest wysoki.")
    elif intellectual < -0.5:
        print("Oj, niski wynik biorytmu intelektualnego. Trzymaj się!")
        next_intellectual = calculate_biorhythm(
            birth_date, today + datetime.timedelta(days=1)
        )[2]
        if next_intellectual > intellectual:
            print("Nie martw się. Jutro będzie lepiej!")


if __name__ == "__main__":
    main()
